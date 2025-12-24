import os
import sys
import shutil
import re
import json
import requests
from marker.models import create_model_dict
from marker.converters.pdf import PdfConverter
from marker.config.parser import ConfigParser
from marker.renderers.markdown import MarkdownRenderer
from marker.settings import settings

# Prevent deadlocks on Windows
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

# Ollama configuration
OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_MODEL = "gpt-oss"  # 20GB model


# ============================================================================
# OLLAMA INTEGRATION
# ============================================================================

def check_ollama_available():
    """Check if Ollama is running and phi3 model is available."""
    try:
        response = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
        if response.status_code == 200:
            models = response.json().get("models", [])
            model_names = [m.get("name", "").split(":")[0] for m in models]
            if OLLAMA_MODEL in model_names or f"{OLLAMA_MODEL}:latest" in [m.get("name", "") for m in models]:
                return True
            print(f"  Warning: {OLLAMA_MODEL} not found. Run: ollama pull {OLLAMA_MODEL}")
            return False
    except Exception as e:
        print(f"  Warning: Ollama not available ({e})")
        print(f"  To enable AI chapter splitting, run: ollama serve")
        return False


def call_ollama(prompt, max_tokens=4096):
    """Call Ollama API with the given prompt."""
    try:
        response = requests.post(
            f"{OLLAMA_BASE_URL}/api/generate",
            json={
                "model": OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "num_predict": max_tokens,
                    "temperature": 0.3,
                }
            },
            timeout=120
        )
        if response.status_code == 200:
            return response.json().get("response", "")
    except Exception as e:
        print(f"  Ollama error: {e}")
    return None


def find_chapter_boundaries_with_ollama(text):
    """
    Use Ollama to identify chapter boundaries in the text.
    Returns list of (line_number, chapter_title) tuples.
    """
    lines = text.split('\n')
    
    # Create a condensed view with line numbers for the first ~200 lines
    # This helps identify the chapter pattern
    preview_lines = []
    for i, line in enumerate(lines[:300]):
        stripped = line.strip()
        if stripped:  # Only include non-empty lines
            preview_lines.append(f"L{i}: {stripped[:100]}")
    
    preview = '\n'.join(preview_lines[:150])  # First 150 non-empty lines
    
    prompt = f"""Analyze this book text and identify chapter boundaries.

Look for patterns like:
- "Chapter 1", "Chapter One", "CHAPTER I"  
- Section headers in ALL CAPS
- Numbered sections "1.", "2." at start of lines
- Part dividers like "PART ONE"

For each chapter you find, provide the LINE NUMBER (L#) and TITLE.

Return as JSON array: [{{"line": 0, "title": "Introduction"}}, {{"line": 45, "title": "Chapter 1 - The Beginning"}}]

Only return the JSON, nothing else.

TEXT:
{preview}"""

    result = call_ollama(prompt, max_tokens=2000)
    
    if result:
        try:
            # Find JSON array in response
            match = re.search(r'\[.*\]', result, re.DOTALL)
            if match:
                boundaries = json.loads(match.group())
                return [(b.get("line", 0), b.get("title", f"Chapter {i+1}")) 
                        for i, b in enumerate(boundaries)]
        except json.JSONDecodeError:
            pass
    
    return [(0, "Full Text")]


def clean_text_with_ollama(text, chunk_size=3000):
    """
    Use Ollama to clean OCR artifacts from text.
    Processes in chunks and returns cleaned text.
    """
    if len(text) <= chunk_size:
        return clean_chunk_with_ollama(text)
    
    # Process in overlapping chunks to avoid cutting words
    cleaned_parts = []
    i = 0
    
    while i < len(text):
        end = min(i + chunk_size, len(text))
        
        # Try to end at a sentence boundary
        if end < len(text):
            # Look for sentence end in last 200 chars
            search_region = text[end-200:end]
            for marker in ['. ', '.\n', '? ', '!\n', '! ', '?\n']:
                last_period = search_region.rfind(marker)
                if last_period != -1:
                    end = end - 200 + last_period + len(marker)
                    break
        
        chunk = text[i:end]
        print(f"    Cleaning chunk {len(cleaned_parts)+1} ({len(chunk)} chars)...")
        
        cleaned = clean_chunk_with_ollama(chunk)
        cleaned_parts.append(cleaned if cleaned else chunk)
        
        i = end
    
    return ''.join(cleaned_parts)


def clean_chunk_with_ollama(text):
    """Clean a single chunk of text using Ollama."""
    prompt = """Fix OCR artifacts in this text. Common issues:
- Split words: "i Pad" → "iPad", "to gether" → "together", "de tachable" → "detachable"
- Merged words: "andthe" → "and the", "inthe" → "in the"
- Random line breaks mid-sentence
- Stray characters or symbols

Keep the original meaning and structure. Return ONLY the cleaned text.

TEXT:
""" + text
    
    result = call_ollama(prompt, max_tokens=len(text) + 500)
    return result if result else text


def split_chapters_with_ollama(full_text, book_name, chapters_folder):
    """
    Use Ollama to intelligently split text into chapters.
    Returns number of chapters created.
    """
    print(f"\n  Using Ollama ({OLLAMA_MODEL}) to identify chapters...")
    
    lines = full_text.split('\n')
    
    # Get chapter boundaries from Ollama
    boundaries = find_chapter_boundaries_with_ollama(full_text)
    print(f"  Found {len(boundaries)} chapters")
    
    if len(boundaries) <= 1:
        # Couldn't find clear boundaries, save as single file
        print("  No clear chapter boundaries found - saving as single file")
        filepath = os.path.join(chapters_folder, f"{book_name}.md")
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(full_text)
        return 1
    
    # Add end boundary
    boundaries.append((len(lines), "END"))
    
    # Split at boundaries
    chapter_count = 0
    for i in range(len(boundaries) - 1):
        start_line, title = boundaries[i]
        end_line = boundaries[i + 1][0]
        
        chapter_text = '\n'.join(lines[start_line:end_line])
        
        if len(chapter_text.strip()) < 100:
            continue
        
        chapter_count += 1
        
        # Create safe filename from title
        safe_title = re.sub(r'[^\w\s-]', '', title).strip().replace(' ', '_')[:50]
        filename = f"Chapter_{chapter_count:02d}_{safe_title}.md"
        
        filepath = os.path.join(chapters_folder, filename)
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(chapter_text)
        print(f"  -> Saved: {filename} ({len(chapter_text)} chars)")
    
    return chapter_count


def process_with_ollama(full_text, book_name, book_folder, chapters_folder):
    """
    Process the converted text with Ollama for chapter splitting and artifact removal.
    """
    print(f"\n{'='*60}")
    print(f"OLLAMA PROCESSING: {book_name}")
    print(f"{'='*60}")
    
    # Step 1: Split into chapters
    chapter_count = split_chapters_with_ollama(full_text, book_name, chapters_folder)
    print(f"\n  Created {chapter_count} chapter files")
    
    # Step 2: Clean artifacts from each chapter
    processed_folder = os.path.join(book_folder, "chapters_processed")
    os.makedirs(processed_folder, exist_ok=True)
    
    chapter_files = sorted([f for f in os.listdir(chapters_folder) if f.endswith('.md')])
    
    print(f"\n  Cleaning OCR artifacts from {len(chapter_files)} files...")
    for filename in chapter_files:
        filepath = os.path.join(chapters_folder, filename)
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()
        
        print(f"  Processing: {filename}")
        cleaned_content = clean_text_with_ollama(content)
        
        processed_path = os.path.join(processed_folder, filename)
        with open(processed_path, "w", encoding="utf-8") as f:
            f.write(cleaned_content)
    
    print(f"\n  Processed chapters saved to: {processed_folder}")
    return chapter_count


# ============================================================================
# MAIN PROCESSING
# ============================================================================

def process_book(pdf_path, base_output_dir, model_dict, use_ollama=True):
    """
    Process a PDF book:
    1. Convert PDF to markdown using marker (normal conversion)
    2. Optionally use Ollama to split into chapters and clean artifacts
    """
    book_name = os.path.splitext(os.path.basename(pdf_path))[0]
    book_folder = os.path.join(base_output_dir, book_name)
    chapters_folder = os.path.join(book_folder, "chapters")
    
    if os.path.exists(book_folder):
        shutil.rmtree(book_folder, ignore_errors=True)
    
    os.makedirs(book_folder, exist_ok=True)
    os.makedirs(chapters_folder, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"Processing: {book_name}")
    print(f"{'='*60}")
    
    # ========================================================================
    # STEP 1: Normal marker PDF conversion
    # ========================================================================
    
    print(f"\n  Converting PDF with marker...")
    
    config = {"output_format": "markdown", "output_dir": book_folder}
    config_parser = ConfigParser(config)
    
    converter = PdfConverter(
        config=config_parser.generate_config_dict(),
        artifact_dict=model_dict,
        processor_list=config_parser.get_processors(),
        renderer=config_parser.get_renderer(),
    )
    
    try:
        rendered = converter(pdf_path)
        
        # Get the full markdown text
        full_text = rendered.markdown if hasattr(rendered, 'markdown') else str(rendered)
        
        # Save the full converted text
        full_text_path = os.path.join(book_folder, f"{book_name}.md")
        with open(full_text_path, "w", encoding="utf-8") as f:
            f.write(full_text)
        
        print(f"  Saved full text: {full_text_path}")
        print(f"  Text length: {len(full_text)} characters")
        
        # ====================================================================
        # STEP 2: Use Ollama for chapter splitting and cleanup (if available)
        # ====================================================================
        
        if use_ollama and check_ollama_available():
            process_with_ollama(full_text, book_name, book_folder, chapters_folder)
        else:
            print(f"\n  Ollama not available - skipping AI chapter splitting")
            print(f"  To enable: 1) ollama serve  2) ollama pull {OLLAMA_MODEL}")
            
            # Save the full text as a single chapter file
            single_chapter = os.path.join(chapters_folder, f"{book_name}.md")
            with open(single_chapter, "w", encoding="utf-8") as f:
                f.write(full_text)
        
        print(f"\n{'='*60}")
        print(f"Finished: {book_name}")
        print(f"{'='*60}")
        print(f"  Output folder: {book_folder}")
        print(f"  Full text: {book_name}.md")
        print(f"  Chapters: {chapters_folder}")
        
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
    
    # Check if Ollama is available
    print("\nChecking Ollama availability...")
    ollama_available = check_ollama_available()
    if ollama_available:
        print(f"  Ollama ready with {OLLAMA_MODEL} model")
    else:
        print(f"  Ollama not available - will do conversion only")
        print(f"  To enable AI features:")
        print(f"    1. Start Ollama: ollama serve")
        print(f"    2. Pull model: ollama pull {OLLAMA_MODEL}")
    
    # Load marker models
    device = settings.TORCH_DEVICE_MODEL
    dtype = settings.MODEL_DTYPE
    print(f"\nUsing device: {device} with dtype: {dtype}")
    
    print("Loading marker models...")
    model_dict = create_model_dict(device=device, dtype=dtype)
    
    # Process each PDF
    for pdf in pdfs:
        pdf_path = os.path.join(input_dir, pdf)
        try:
            process_book(pdf_path, output_dir, model_dict, use_ollama=ollama_available)
        except Exception as e:
            print(f"Error processing {pdf}: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()
