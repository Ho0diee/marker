import os
import sys
import shutil
import re
import torch
import requests
from marker.models import create_model_dict
from marker.converters.pdf import PdfConverter
from marker.config.parser import ConfigParser
from marker.renderers.markdown import Markdownify, cleanup_text
from marker.settings import settings
import wordsegment

# Prevent deadlocks on Windows
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

# Initialize wordsegment
wordsegment.load()


# ============================================================================
# BOOK METADATA LOOKUP
# ============================================================================

def extract_book_info_from_filename(filename):
    """Extract potential book title and author from filename."""
    # Remove extension and common patterns
    name = os.path.splitext(filename)[0]
    # Remove common suffixes like -1, _final, .pdf-1, etc.
    name = re.sub(r'\.pdf[-_]?\d*$', '', name, flags=re.I)
    name = re.sub(r'[-_]?\d+$', '', name)
    name = re.sub(r'[-_]?(final|copy|scan|ocr|pdf)$', '', name, flags=re.I)
    # Replace separators with spaces
    name = re.sub(r'[-_]+', ' ', name)
    
    # Try to detect author name patterns like "authorYYYYtitle" or "author YYYY title"
    # e.g., "isaacson2011stevejobs" -> "isaacson 2011 steve jobs"
    name = re.sub(r'(\d{4})', r' \1 ', name)  # Add spaces around years
    name = re.sub(r'([a-z])([A-Z])', r'\1 \2', name)  # Split camelCase
    
    # Clean up multiple spaces
    name = ' '.join(name.split())
    return name.strip()


def extract_book_title_from_pdf(rendered):
    """Try to extract book title from the PDF content itself."""
    # Look at first few blocks for title-like content
    for block in rendered.blocks[:20]:
        if block.block_type == "SectionHeader":
            title = re.sub('<[^<]+?>', '', block.html).strip()
            # Skip obvious non-titles
            skip_words = ['contents', 'copyright', 'dedication', 'preface', 'chapter', 'introduction']
            if len(title) > 3 and not any(w in title.lower() for w in skip_words):
                # This might be the book title
                return title
    return None


def search_book_info_google_books(query):
    """Search Google Books API for book info."""
    try:
        url = f"https://www.googleapis.com/books/v1/volumes?q={requests.utils.quote(query)}&maxResults=5"
        response = requests.get(url, timeout=10)
        data = response.json()
        
        if 'items' in data:
            for item in data['items']:
                info = item.get('volumeInfo', {})
                title = info.get('title', '')
                authors = info.get('authors', [])
                description = info.get('description', '').lower()
                
                # Look for chapter count in description
                patterns = [
                    r'(\d+)\s*chapters',
                    r'chapters[:\s]+(\d+)',
                    r'(\d+)\s+chapter',
                ]
                
                for pattern in patterns:
                    matches = re.findall(pattern, description)
                    for match in matches:
                        count = int(match)
                        if 5 <= count <= 200:
                            return {
                                'title': title,
                                'authors': authors,
                                'chapters': count,
                                'source': 'google_books'
                            }
                
                # Return book info even without chapter count
                if title:
                    return {
                        'title': title,
                        'authors': authors,
                        'chapters': None,
                        'source': 'google_books'
                    }
    except Exception as e:
        print(f"    Google Books API failed: {e}")
    return None


def search_book_info_openlibrary(query):
    """Search Open Library API for book info."""
    try:
        url = f"https://openlibrary.org/search.json?q={requests.utils.quote(query)}&limit=5"
        response = requests.get(url, timeout=10)
        data = response.json()
        
        if 'docs' in data and data['docs']:
            for doc in data['docs']:
                title = doc.get('title', '')
                authors = doc.get('author_name', [])
                
                # Open Library sometimes has number_of_pages, but not chapters
                # We can use this as a rough heuristic (avg 15-25 pages per chapter)
                num_pages = doc.get('number_of_pages_median')
                
                if title:
                    return {
                        'title': title,
                        'authors': authors,
                        'pages': num_pages,
                        'source': 'openlibrary'
                    }
    except Exception as e:
        print(f"    Open Library API failed: {e}")
    return None


def search_wikipedia_chapters(book_title, author=None):
    """Search Wikipedia for chapter count or page count."""
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) BookConverter/1.0'}
    
    try:
        # Try searching Wikipedia for the book
        search_term = f"{book_title} book" if not author else f"{book_title} {author}"
        url = f"https://en.wikipedia.org/w/api.php?action=query&list=search&srsearch={requests.utils.quote(search_term)}&format=json"
        
        response = requests.get(url, headers=headers, timeout=10)
        data = response.json()
        
        if 'query' in data and data['query'].get('search'):
            # Get the first result's page
            page_title = data['query']['search'][0]['title']
            
            # Get page wikitext (contains page count in infobox)
            wikitext_url = f"https://en.wikipedia.org/w/api.php?action=parse&page={requests.utils.quote(page_title)}&prop=wikitext&format=json"
            wikitext_response = requests.get(wikitext_url, headers=headers, timeout=10)
            wikitext_data = wikitext_response.json()
            
            wikitext = wikitext_data.get('parse', {}).get('wikitext', {}).get('*', '')
            
            # Look for chapter count first
            chapter_patterns = [
                r'(\d+)\s*chapters',
                r'chapters\s*=\s*(\d+)',
            ]
            
            for pattern in chapter_patterns:
                matches = re.findall(pattern, wikitext.lower())
                for match in matches:
                    count = int(match)
                    if 5 <= count <= 200:
                        return {'chapters': count, 'source': 'wikipedia'}
            
            # Look for page count in infobox
            page_patterns = [
                r'\|\s*pages\s*=\s*(\d+)',
                r'(\d+)\s*pp\.',
                r'(\d+)\s*pages',
            ]
            
            for pattern in page_patterns:
                matches = re.findall(pattern, wikitext)
                for match in matches:
                    pages = int(match)
                    if 100 <= pages <= 2000:
                        # Estimate chapters from pages (avg 15 pages per chapter for biographies)
                        estimated_chapters = round(pages / 15)
                        return {'pages': pages, 'estimated_chapters': estimated_chapters, 'source': 'wikipedia'}
            
    except Exception as e:
        print(f"    Wikipedia search failed: {e}")
    return None


def search_book_chapter_count(book_name, pdf_title=None):
    """
    Search multiple sources for expected chapter count of a book.
    Returns (chapter_count, confidence) or (None, 0) if not found.
    """
    print(f"  Searching book databases...")
    
    # Build search queries
    queries_to_try = []
    
    if pdf_title:
        queries_to_try.append(pdf_title)
    
    # Parse filename for author/title
    # Pattern: "authorYYYYtitle" -> "author title"
    year_match = re.search(r'(19|20)\d{2}', book_name)
    if year_match:
        parts = re.split(r'(19|20)\d{2}', book_name)
        if len(parts) >= 2:
            author = parts[0].strip()
            title_parts = [p.strip() for p in parts[1:] if p.strip() and not re.match(r'^\d+$', p)]
            title = ' '.join(title_parts)
            if author and title:
                queries_to_try.append(f"{title} {author}")
                queries_to_try.append(title)
    
    queries_to_try.append(book_name)
    
    # Remove duplicates while preserving order
    seen = set()
    queries_to_try = [q for q in queries_to_try if not (q in seen or seen.add(q))]
    
    print(f"    Queries: {queries_to_try[:3]}")
    
    book_info = None
    chapter_count = None
    
    # Try each query
    for query in queries_to_try[:3]:
        print(f"    Trying: '{query}'")
        
        # Try Google Books
        info = search_book_info_google_books(query)
        if info:
            print(f"    Found on Google Books: '{info.get('title')}' by {info.get('authors', ['Unknown'])}")
            book_info = info
            if info.get('chapters'):
                chapter_count = info['chapters']
                print(f"    Chapter count from description: {chapter_count}")
                return chapter_count, 2
            
            # Try Wikipedia with the confirmed title
            wiki_chapters = search_wikipedia_chapters(info['title'], info.get('authors', [None])[0] if info.get('authors') else None)
            if wiki_chapters:
                print(f"    Chapter count from Wikipedia: {wiki_chapters}")
                return wiki_chapters, 3
        
        # Try Open Library
        if not book_info:
            info = search_book_info_openlibrary(query)
            if info:
                print(f"    Found on Open Library: '{info.get('title')}'")
                book_info = info
                
                # Estimate chapters from pages (rough: 1 chapter per 10-20 pages)
                if info.get('pages'):
                    estimated = info['pages'] // 15
                    if 5 <= estimated <= 100:
                        print(f"    Estimated from {info['pages']} pages: ~{estimated} chapters")
                        return estimated, 1  # Low confidence estimate
    
    # If we found a title but no chapter count, try Wikipedia directly
    if book_info and not chapter_count:
        wiki_result = search_wikipedia_chapters(book_info['title'])
        if wiki_result:
            if wiki_result.get('chapters'):
                print(f"    Chapter count from Wikipedia: {wiki_result['chapters']}")
                return wiki_result['chapters'], 3
            elif wiki_result.get('estimated_chapters'):
                print(f"    Estimated from {wiki_result['pages']} pages (Wikipedia): ~{wiki_result['estimated_chapters']} chapters")
                return wiki_result['estimated_chapters'], 1
    
    return None, 0


def detect_chapters_with_strategy(rendered, md_converter, strategy="h1_only"):
    """
    Detect chapters using different strategies.
    Returns list of (title, text) tuples.
    """
    ALLOWED_BLOCKS = {"Text", "SectionHeader", "List", "Equation"}
    chapters = []
    current_title = "Front_Matter"
    current_text = ""
    pending_chapter_num = None  # For chapter_pattern: track chapter number, look for title in next blocks
    
    # Chapter pattern regex for detecting explicit chapter markers
    chapter_pattern = re.compile(
        r'^(?:\*\*)?(?:Chapter|CHAPTER)\s+(\d+|[IVXLC]+|One|Two|Three|Four|Five|Six|Seven|Eight|Nine|Ten|Eleven|Twelve|Thirteen|Fourteen|Fifteen|Sixteen|Seventeen|Eighteen|Nineteen|Twenty)(?:\*\*)?(?:\s*[-:.]?\s*(.*))?$',
        re.IGNORECASE
    )
    
    for block in rendered.blocks:
        if block.block_type not in ALLOWED_BLOCKS:
            continue
        
        is_chapter_break = False
        block_text = re.sub('<[^<]+?>', '', block.html).strip()
        chapter_title_from_pattern = None
        
        if strategy == "chapter_pattern":
            # Look for explicit chapter markers in any block type
            match = chapter_pattern.search(block_text)
            if match:
                is_chapter_break = True
                chapter_num = match.group(1)
                # Check if there's a title on the same line (after the number)
                title_part = match.group(2).strip() if match.group(2) else ""
                if title_part and len(title_part) > 3:
                    chapter_title_from_pattern = f"Chapter {chapter_num} - {title_part}"
                else:
                    # Title might be in subsequent blocks - we'll use chapter number for now
                    # and capture the title from the next significant text block
                    pending_chapter_num = chapter_num
            elif pending_chapter_num and block.block_type == "SectionHeader":
                # This header block might be the chapter title
                title_candidate = block_text.strip()
                # Skip if it's another chapter marker or very short
                if title_candidate and len(title_candidate) > 3 and not chapter_pattern.search(title_candidate):
                    current_title = f"Chapter {pending_chapter_num} - {title_candidate}"
                    pending_chapter_num = None
        elif block.block_type == "SectionHeader":
            hierarchy = block.section_hierarchy or {}
            
            if strategy == "h1_only":
                # Only H1 headers
                if hierarchy.get(1) == block.id or hierarchy.get("1") == block.id or \
                   re.search(r'<h1[^>]*>', block.html, re.I):
                    is_chapter_break = True
                    
            elif strategy == "h1_h2":
                # H1 and H2 headers
                if hierarchy.get(1) == block.id or hierarchy.get("1") == block.id or \
                   hierarchy.get(2) == block.id or hierarchy.get("2") == block.id or \
                   re.search(r'<h[12][^>]*>', block.html, re.I):
                    is_chapter_break = True
                    
            elif strategy == "h1_h2_h3":
                # H1, H2, and H3 headers
                if hierarchy.get(1) == block.id or hierarchy.get("1") == block.id or \
                   hierarchy.get(2) == block.id or hierarchy.get("2") == block.id or \
                   hierarchy.get(3) == block.id or hierarchy.get("3") == block.id or \
                   re.search(r'<h[123][^>]*>', block.html, re.I):
                    is_chapter_break = True
        
        block_md = md_converter.convert(block.html)
        
        if is_chapter_break:
            if current_text.strip():
                chapters.append((current_title, current_text))
            
            if chapter_title_from_pattern:
                current_title = chapter_title_from_pattern
            elif pending_chapter_num:
                # Will be updated when we find the title in next block
                current_title = f"Chapter {pending_chapter_num}"
            else:
                title_text = re.sub('<[^<]+?>', '', block.html).strip()
                current_title = " ".join(title_text.split()) or "Chapter"
            current_text = block_md + "\n\n"
        else:
            current_text += block_md + "\n\n"
    
    # Don't forget final chapter
    if current_text.strip():
        chapters.append((current_title, current_text))
    
    return chapters


def choose_best_strategy(rendered, md_converter, expected_count=None):
    """
    Try different detection strategies and choose the best one.
    If expected_count is provided, choose the strategy closest to it.
    """
    strategies = ["h1_only", "h1_h2", "h1_h2_h3", "chapter_pattern"]
    results = {}
    
    for strategy in strategies:
        chapters = detect_chapters_with_strategy(rendered, md_converter, strategy)
        # Filter out obvious non-chapters
        real_chapters = [c for c in chapters if not is_skip_title(c[0])]
        results[strategy] = len(real_chapters)
        print(f"    {strategy}: detected {len(real_chapters)} chapters")
    
    # Extract the chapter count if expected_count is a dict
    if isinstance(expected_count, dict):
        expected_count = expected_count.get('estimated_chapters')
    
    # If no expected count, try to extract from TOC
    if not expected_count:
        toc_count, toc_titles = extract_toc_chapter_count(rendered, md_converter)
        if toc_count and toc_count >= 5:
            expected_count = toc_count
            print(f"    (Extracted {toc_count} chapters from Table of Contents)")
    
    if expected_count:
        # Choose strategy closest to expected
        best_strategy = min(strategies, key=lambda s: abs(results[s] - expected_count))
        best_count = results[best_strategy]
        
        # If best strategy is still wildly off (more than 3x or less than 0.33x expected),
        # try smarter fallback
        if best_strategy != "chapter_pattern":
            ratio = best_count / expected_count if expected_count > 0 else 0
            pattern_count = results["chapter_pattern"]
            pattern_ratio = pattern_count / expected_count if expected_count > 0 else 0
            
            if (ratio > 3 or ratio < 0.33):
                # Header detection is way off
                if (0.5 <= pattern_ratio <= 2.0):
                    print(f"  -> Header-based detection is way off ({best_count} vs expected {expected_count})")
                    print(f"  -> Switching to chapter_pattern strategy ({pattern_count} chapters)")
                    return "chapter_pattern"
                else:
                    # Even chapter_pattern doesn't work well, use expected count as guide
                    # to pick the least-bad option
                    print(f"  -> Warning: All detection strategies are off from expected {expected_count}")
                    print(f"  -> Using {best_strategy} ({best_count} chapters) as best available option")
        
        print(f"  -> Best match for {expected_count} chapters: {best_strategy} ({best_count} chapters)")
        return best_strategy
    else:
        # No expected count - use heuristics
        # Prefer counts in the "reasonable" range (15-40 for most books)
        reasonable_range = lambda x: 15 <= x <= 40
        
        # First check if any strategy gives a reasonable count
        for strategy in strategies:
            if reasonable_range(results[strategy]):
                print(f"  -> Using {strategy} ({results[strategy]} chapters) - reasonable count")
                return strategy
        
        # Fallback: prefer h1_only if it gives count under 50
        if results["h1_only"] <= 50:
            return "h1_only"
        elif results["h1_h2"] <= 50:
            return "h1_h2"
        elif 5 <= results["chapter_pattern"] <= 100:
            return "chapter_pattern"
        else:
            # Last resort - pick the one with smallest count over 10
            valid = {s: c for s, c in results.items() if c >= 10}
            if valid:
                return min(valid, key=valid.get)
            return "h1_only"


def is_skip_title(title):
    """Check if a title should be skipped (front/back matter, part dividers)."""
    SKIP_KEYWORDS = [
        "CONTENTS", "TABLE OF CONTENTS", "DEDICATION", "ACKNOWLEDGEMENTS",
        "TITLE PAGE", "COPYRIGHT", "ABOUT THE AUTHOR", "ALSO BY", "PRAISE",
        "MAPS", "INDEX", "NOTES", "BIBLIOGRAPHY", "CREDITS", "CHARACTERS",
        "FRONT_MATTER", "EPIGRAPH", "FOREWORD", "PREFACE",
        # Part dividers
        "THE FUNDAMENTALS", "THE 1ST LAW", "THE 2ND LAW", "THE 3RD LAW", 
        "THE 4TH LAW", "THE 5TH LAW", "ADVANCED TACTICS",
        "PART ONE", "PART TWO", "PART THREE", "PART FOUR", "PART FIVE",
        "PART I", "PART II", "PART III", "PART IV", "PART V",
        "PART 1", "PART 2", "PART 3", "PART 4", "PART 5",
        # Common section breaks
        "APPENDIX", "CHAPTER SUMMARY", "WHAT'S NEXT",
        # Front matter patterns
        "ISBN", "EBOOK", "E-BOOK",
        # Back matter patterns  
        "WHAT SHOULD YOU READ", "LITTLE LESSONS", "HOW TO APPLY THESE IDEAS",
        "FURTHER READING", "RECOMMENDED READING", "SUGGESTED READING",
        "ACKNOWLEDGMENTS", "ACKNOWLEDGEMENTS",
    ]
    title_upper = title.upper().strip()
    
    # Skip empty or very short titles
    if len(title_upper) < 3:
        return True
    
    # Skip if it matches any keyword
    for kw in SKIP_KEYWORDS:
        if kw in title_upper:
            return True
    
    # Skip generic "Chapter" without any descriptive title
    if title_upper == "CHAPTER" or re.match(r'^CHAPTER\s*\d*$', title_upper):
        return True
    
    return False


def is_duplicate_or_notes_chapter(title, existing_titles):
    """
    Check if a chapter title is a duplicate or from Notes section.
    Returns True if this chapter should be skipped.
    """
    title_normalized = title.upper().strip()
    
    # Check for exact duplicates or very similar titles
    for existing in existing_titles:
        existing_normalized = existing.upper().strip()
        if title_normalized == existing_normalized:
            return True
        # Check if one is a substring of the other (e.g., "INTRODUCTION" vs "Introduction")
        if len(title_normalized) > 5 and len(existing_normalized) > 5:
            if title_normalized in existing_normalized or existing_normalized in title_normalized:
                return True
    
    return False


def extract_toc_chapter_count(rendered, md_converter):
    """
    Try to extract chapter count from Table of Contents.
    Returns (count, chapter_titles) or (None, []) if not found.
    """
    toc_started = False
    toc_content = []
    chapter_titles = []
    
    for block in rendered.blocks:
        block_text = re.sub('<[^<]+?>', '', block.html).strip()
        block_md = md_converter.convert(block.html)
        
        # Look for TOC start
        if "CONTENTS" in block_text.upper() or "TABLE OF CONTENTS" in block_text.upper():
            toc_started = True
            continue
        
        if toc_started:
            # TOC usually ends when we hit Introduction or Chapter 1 content (longer text blocks)
            if len(block_text) > 500:
                break
                
            toc_content.append(block_text)
            
            # Look for numbered chapter entries like "1 The Surprising Power" or "Chapter 1"
            # Also look for common chapter title patterns
            numbered_match = re.match(r'^(\d{1,2})\s+(.+)', block_text)
            chapter_match = re.match(r'^(?:Chapter|CHAPTER)\s+(\d+)', block_text, re.I)
            
            if numbered_match and len(numbered_match.group(2)) > 5:
                chapter_titles.append(numbered_match.group(2).strip())
            elif chapter_match:
                chapter_titles.append(block_text)
    
    if chapter_titles:
        return len(chapter_titles), chapter_titles
    return None, []


# ============================================================================
# POST-PROCESSING FUNCTIONS (Applied in second pass only)
# ============================================================================

def fix_split_words(text):
    """
    Fix words that were incorrectly split during OCR/conversion.
    Examples: "i Pad" -> "iPad", "de tachable" -> "detachable"
    """
    # Common product names and proper nouns with incorrect spacing
    product_fixes = {
        r'\bi\s*Pad\b': 'iPad',
        r'\bi\s*Phone\b': 'iPhone',
        r'\bi\s*Pod\b': 'iPod',
        r'\bi\s*Mac\b': 'iMac',
        r'\bi\s*Tunes\b': 'iTunes',
        r'\bi\s*Cloud\b': 'iCloud',
        r'\bi\s*OS\b': 'iOS',
        r'\bi\s*Work\b': 'iWork',
        r'\bi\s*Life\b': 'iLife',
        r'\bi\s*Movie\b': 'iMovie',
        r'\bi\s*Photo\b': 'iPhoto',
        r'\bMac\s*Book\b': 'MacBook',
        r'\bPower\s*Book\b': 'PowerBook',
        r'\bFace\s*Time\b': 'FaceTime',
        r'\bApp\s*Store\b': 'App Store',
        r'\bJony\s+I\s+ve\b': 'Jony Ive',
        r'\bJony\s+I\s*ve\b': 'Jony Ive',
        r'\bI\s+ve\b(?!\s+been|\s+got|\s+had|\s+always|\s+never)': 'Ive',
    }
    
    for pattern, replacement in product_fixes.items():
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    
    # Fix single letter + rest of word splits: "de tachable" -> "detachable"
    def fix_split_word(match):
        prefix = match.group(1)
        suffix = match.group(2)
        combined = prefix + suffix
        segments = wordsegment.segment(combined.lower())
        if len(segments) == 1:
            return combined
        return match.group(0)
    
    # Fix patterns like "de tachable", "jus tlook"
    text = re.sub(r'\b([a-z]{1,3})\s+([a-z]{3,})\b', fix_split_word, text)
    
    # Fix merged words using wordsegment
    def fix_merged_words_smart(match):
        word = match.group(0)
        if len(word) > 12:
            segments = wordsegment.segment(word.lower())
            if len(segments) > 1:
                result = ' '.join(segments)
                if word[0].isupper():
                    result = result[0].upper() + result[1:]
                return result
        return word
    
    text = re.sub(r'\b[a-z]{13,}\b', fix_merged_words_smart, text, flags=re.IGNORECASE)
    
    return text


def fix_merged_words(text):
    """Fix common word merging issues from OCR."""
    # 1. Fix drop-caps (e.g., "I taly" -> "Italy", "A n" -> "An")
    text = re.sub(r'\b([A-Z])\s+([a-z]{2,})\b', r'\1\2', text)
    
    # 2. Fix CamelCase and Number-Letter merges
    text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)
    text = re.sub(r'([0-9])([a-zA-Z])', r'\1 \2', text)
    text = re.sub(r'([a-zA-Z])([0-9])', r'\1 \2', text)

    # 3. Fix common small words stuck to others
    common_merges = {
        r'\bwiththe\b': 'with the',
        r'\bbetthathe\b': 'bet that he',
        r'\badate\b': 'a date',
        r'\byearslater\b': 'years later',
        r'\bschoolprank\b': 'school prank',
        r'\bItwasn\'t\b': "It wasn't",
        r'\bofhis\b': 'of his',
        r'\bandthe\b': 'and the',
        r'\binthe\b': 'in the',
        r'\bhewas\b': 'he was',
        r'\btohis\b': 'to his',
        r'\bforhis\b': 'for his',
        r'\bfromthe\b': 'from the',
        r'\bthathe\b': 'that he',
        r'\bhadbeen\b': 'had been',
        r'\bshewas\b': 'she was',
        r'\bonthe\b': 'on the',
        r'\batthe\b': 'at the',
        r'\bwitha\b': 'with a',
        r'\bintoa\b': 'into a',
        r'\btherewas\b': 'there was',
        r'\btherewasa\b': 'there was a',
        r'\bdefinitelywanted\b': 'definitely wanted',
        r'\ballofits\b': 'all of its',
        r'\bbegantoopen\b': 'began to open',
        r'\bjustlook\b': 'just look',
        r'\bjus\s*tlook\b': 'just look',
    }
    for pattern, replacement in common_merges.items():
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    
    # 4. Fix "I" merges
    text = re.sub(r'(\s|^)I([a-z]{2,})', r'\1I \2', text)
    
    # 5. Fix specific "I <space> <letter>" artifacts
    text = re.sub(r'\bI\s+t\b', 'It', text)
    text = re.sub(r'\bI\s+n\b', 'In', text)
    
    return text


def apply_all_text_fixes(text):
    """Apply all post-processing text fixes."""
    text = fix_split_words(text)
    text = fix_merged_words(text)
    return text


# ============================================================================
# BASIC TEXT UTILITIES (Used in first pass)
# ============================================================================

def strip_markdown(text):
    """Remove markdown formatting for clean text output."""
    # Remove images and links
    text = re.sub(r'!\[.*?\]\(.*?\)', '', text)
    text = re.sub(r'\[(.*?)\]\(.*?\)', r'\1', text)
    # Remove bold/italic
    text = re.sub(r'(\*\*|__)(.*?)\1', r'\2', text)
    text = re.sub(r'(\*|_)(.*?)\1', r'\2', text)
    # Remove headers
    text = re.sub(r'^#+\s+', '', text, flags=re.MULTILINE)
    # Remove horizontal rules
    text = re.sub(r'^\s*[-*_]{3,}\s*$', '', text, flags=re.MULTILINE)
    # Clean up escaped characters
    text = text.replace('\\*', '*').replace('\\_', '_').replace('\\-', '-').replace('\\.', '.')
    # Remove excessive dots/dashes
    text = re.sub(r'\.{4,}', '', text)
    text = re.sub(r'-{4,}', '', text)
    return text.strip()


def clean_chapter_title(title):
    """Clean up chapter title for filename."""
    title = re.sub(r'([a-z])([A-Z])', r'\1 \2', title)
    title = re.sub(r'^(Chapter|CHAPTER)\s+([0-9]+|[A-Z]+)\s*[:.-]?\s*', '', title, flags=re.IGNORECASE)
    title = title.strip(' :.-_')
    return title


def get_word_count(text):
    """Get approximate word count of text."""
    words = re.findall(r'\b\w+\b', text)
    return len(words)


# ============================================================================
# CHAPTER DETECTION (First pass - pure AI-based detection)
# ============================================================================

def should_skip_chapter_first_pass(title, text, skip_keywords):
    """
    First pass skip logic - only skip obvious non-chapters.
    Keep the AI's chapter detection as clean as possible.
    """
    title_upper = title.upper()
    
    # Skip obvious front/back matter by title
    if title == "Front_Matter":
        return True
    if any(kw in title_upper for kw in skip_keywords):
        return True
    
    # Skip if it looks like a table of contents
    clean_text = text.strip()
    first_chars = clean_text[:1500].upper()
    if "CONTENTS" in first_chars or "TABLE OF CONTENTS" in first_chars:
        if len(re.findall(r'\d+', first_chars)) > 10:
            return True
    
    return False


def save_chapter_first_pass(title, text, count, folder, skip_keywords):
    """Save chapter in first pass without heavy text processing."""
    if should_skip_chapter_first_pass(title, text, skip_keywords):
        print(f"  (Skipping non-chapter: {title[:30]}...)")
        return False

    display_title = clean_chapter_title(title)
    safe_title = re.sub(r'[^\w\s-]', '', display_title).strip().replace(' ', '_')
    
    # Light cleanup only - no word fixing
    clean_text = strip_markdown(cleanup_text(text))
    
    filename = f"Chapter {count+1:02d} - {safe_title}.md"
    filepath = os.path.join(folder, filename)
    
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(clean_text)
    print(f"  -> Saved: {filename}")
    return True


# ============================================================================
# POST-PROCESSING (Second pass - fixes applied to copies)
# ============================================================================

def post_process_chapters(chapters_folder, processed_folder, min_first_chapter_words=500):
    """
    Second pass: Create processed copies with all fixes applied.
    
    - Skip first chapter if word count is too low (likely front matter)
    - Apply text fixes for split/merged words
    """
    if not os.path.exists(chapters_folder):
        print(f"  No chapters folder found: {chapters_folder}")
        return
    
    os.makedirs(processed_folder, exist_ok=True)
    
    # Get all chapter files sorted
    chapter_files = sorted([f for f in os.listdir(chapters_folder) if f.endswith('.md')])
    
    if not chapter_files:
        print("  No chapter files found to process.")
        return
    
    print(f"\n  Post-processing {len(chapter_files)} chapters...")
    
    # Check first chapter word count
    first_chapter_path = os.path.join(chapters_folder, chapter_files[0])
    with open(first_chapter_path, "r", encoding="utf-8") as f:
        first_content = f.read()
    
    first_word_count = get_word_count(first_content)
    skip_first = first_word_count < min_first_chapter_words
    
    if skip_first:
        print(f"  Skipping first chapter '{chapter_files[0]}' (only {first_word_count} words)")
        chapter_files = chapter_files[1:]
    
    # Process remaining chapters
    new_chapter_num = 1
    for old_filename in chapter_files:
        old_path = os.path.join(chapters_folder, old_filename)
        
        with open(old_path, "r", encoding="utf-8") as f:
            content = f.read()
        
        # Apply all text fixes
        fixed_content = apply_all_text_fixes(content)
        
        # Create new filename with updated numbering
        match = re.match(r'Chapter \d+ - (.+)\.md', old_filename)
        if match:
            title_part = match.group(1)
        else:
            title_part = old_filename.replace('.md', '')
        
        new_filename = f"Chapter {new_chapter_num:02d} - {title_part}.md"
        new_path = os.path.join(processed_folder, new_filename)
        
        with open(new_path, "w", encoding="utf-8") as f:
            f.write(fixed_content)
        
        print(f"  -> Processed: {new_filename}")
        new_chapter_num += 1
    
    print(f"  Post-processing complete. {new_chapter_num - 1} chapters saved to {processed_folder}")


# ============================================================================
# MAIN PROCESSING
# ============================================================================

def process_book(pdf_path, base_output_dir, model_dict):
    book_name = os.path.splitext(os.path.basename(pdf_path))[0]
    book_folder = os.path.join(base_output_dir, book_name)
    chapters_folder = os.path.join(book_folder, "chapters")
    processed_folder = os.path.join(book_folder, "chapters_processed")
    
    if os.path.exists(book_folder):
        shutil.rmtree(book_folder, ignore_errors=True)
    
    os.makedirs(chapters_folder, exist_ok=True)
    
    # ========================================================================
    # STEP 0: Search for expected chapter count online
    # ========================================================================
    
    print(f"\n{'='*60}")
    print(f"Processing: {book_name}")
    print(f"{'='*60}")
    
    # ========================================================================
    # STEP 1: Convert PDF first (so we can extract title)
    # ========================================================================
    
    print(f"\n  Converting PDF...")
    
    config = {"output_format": "chunks", "output_dir": book_folder}
    config_parser = ConfigParser(config)
    converter = PdfConverter(
        config=config_parser.generate_config_dict(),
        artifact_dict=model_dict,
        processor_list=config_parser.get_processors(),
        renderer="marker.renderers.chunk.ChunkRenderer",
    )
    
    md_converter = Markdownify(
        paginate_output=False,
        page_separator="",
        heading_style="ATX",
        bullets="-",
        escape_misc=False,
        escape_underscores=True,
        escape_asterisks=True,
        escape_dollars=True,
        sub_symbol="<sub>",
        sup_symbol="<sup>",
        inline_math_delimiters=("$", "$"),
        block_math_delimiters=("$$", "$$"),
        html_tables_in_markdown=False
    )

    try:
        rendered = converter(pdf_path)
        
        # ====================================================================
        # STEP 2: Search for expected chapter count online
        # ====================================================================
        
        book_search_name = extract_book_info_from_filename(os.path.basename(pdf_path))
        pdf_title = extract_book_title_from_pdf(rendered)
        
        print(f"\n  Extracted filename: '{book_search_name}'")
        if pdf_title:
            print(f"  Extracted PDF title: '{pdf_title}'")
        
        print(f"\n  Searching online for chapter count...")
        expected_chapters, confidence = search_book_chapter_count(book_search_name, pdf_title)
        
        if expected_chapters:
            print(f"  Found expected chapter count: {expected_chapters} (confidence: {confidence})")
        else:
            print(f"  Could not find chapter count online, will use heuristics")
        
        # ====================================================================
        # STEP 3: Choose best chapter detection strategy
        # ====================================================================
        
        print(f"\n  Testing chapter detection strategies...")
        best_strategy = choose_best_strategy(rendered, md_converter, expected_chapters)
        
        # ====================================================================
        # STEP 4: Extract chapters using chosen strategy
        # ====================================================================
        
        print(f"\n  Extracting chapters using '{best_strategy}' strategy...")
        all_chapters = detect_chapters_with_strategy(rendered, md_converter, best_strategy)
        
        # Filter and save chapters
        real_chapter_count = 0
        full_text_md = ""
        saved_titles = []  # Track saved titles to detect duplicates
        
        for title, text in all_chapters:
            full_text_md += text
            
            if is_skip_title(title):
                print(f"  (Skipping non-chapter: {title[:30]}...)")
                continue
            
            # Check for duplicate titles (often from Notes section referencing earlier chapters)
            if is_duplicate_or_notes_chapter(title, saved_titles):
                print(f"  (Skipping duplicate: {title[:30]}...)")
                continue
            
            # Check minimum length
            if len(text.strip()) < 500:
                print(f"  (Skipping short section: {title[:30]}...)")
                continue
            
            display_title = clean_chapter_title(title)
            safe_title = re.sub(r'[^\w\s-]', '', display_title).strip().replace(' ', '_')
            clean_text = strip_markdown(cleanup_text(text))
            
            filename = f"Chapter {real_chapter_count+1:02d} - {safe_title}.md"
            filepath = os.path.join(chapters_folder, filename)
            
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(clean_text)
            print(f"  -> Saved: {filename}")
            saved_titles.append(title)  # Track this title
            real_chapter_count += 1

        # Save full text
        full_text_md = cleanup_text(full_text_md)
        with open(os.path.join(book_folder, f"{book_name}.md"), "w", encoding="utf-8") as f:
            f.write(full_text_md)
            
        print(f"\n  First pass complete: {real_chapter_count} chapters saved")
        
        # ====================================================================
        # STEP 5: Post-processing with fixes
        # ====================================================================
        
        print(f"\n{'='*60}")
        print(f"POST-PROCESSING: Applying fixes to {book_name}")
        print(f"{'='*60}")
        
        post_process_chapters(chapters_folder, processed_folder, min_first_chapter_words=500)
        
        # Also create a fixed version of the full text
        fixed_full_text = apply_all_text_fixes(full_text_md)
        with open(os.path.join(book_folder, f"{book_name}_processed.md"), "w", encoding="utf-8") as f:
            f.write(fixed_full_text)
        
        print(f"\nFinished {book_name}!")
        print(f"  - Raw chapters: {chapters_folder}")
        print(f"  - Processed chapters: {processed_folder}")
                
    except Exception as e:
        print(f"Error processing {book_name}: {e}")
        import traceback
        traceback.print_exc()


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    input_dir = os.path.join(base_dir, "upload books")
    output_dir = os.path.join(base_dir, "conversion_results")
    
    if not os.path.exists(input_dir):
        os.makedirs(input_dir)
        print(f"Created input folder: {input_dir}. Please place your PDFs there.")
        return

    pdfs = [f for f in os.listdir(input_dir) if f.lower().endswith(".pdf")]
    if not pdfs:
        print(f"No PDFs found in {input_dir}")
        return

    print(f"Found {len(pdfs)} books to process.")
    device = settings.TORCH_DEVICE_MODEL
    dtype = settings.MODEL_DTYPE
    print(f"Using device: {device} with dtype: {dtype}")
    
    print("Loading models...")
    model_dict = create_model_dict(device=device, dtype=dtype)

    for pdf in pdfs:
        pdf_path = os.path.join(input_dir, pdf)
        try:
            process_book(pdf_path, output_dir, model_dict)
        except Exception as e:
            print(f"Error processing {pdf}: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()
