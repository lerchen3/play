import requests
import json
import time
import os
import glob
from io import BytesIO
import zipfile
try:
    from PyPDF2 import PdfReader, PdfWriter
except ImportError:
    print("PyPDF2 not found. Please install it: pip install pypdf2")
    # Consider exiting if PyPDF2 is critical
    # For now, script will fail later if it's not installed and used.
import pandas as pd

# --- Configuration ---
# Replace with your actual App ID and App Key
APP_ID = "lerchenorg_10192b_34c551"
APP_KEY = "3c3768a079634a0ef0e88ce1217c1529a35a956b326aac354ce5a20863a7336d"

# Directory containing your PDF files
PDF_DIRECTORY = "./mypdfs/"  # Make sure this directory exists
# Directory to save the results
OUTPUT_DIRECTORY = "./pdf_latex_results/" # Changed output directory name
OUTPUT_CSV_FILENAME = "extracted_latex_from_mathpix.csv"

# API Endpoints
SUBMIT_PDF_URL = "https://api.mathpix.com/v3/pdf"

# Mathpix conversion options for .tex output
MATHPIX_OPTIONS = {
    "conversion_formats": {"tex.zip": True},
    "math_inline_delimiters": ["$", "$"], # Standard LaTeX delimiters
    "math_display_delimiters": ["$$", "$$"]
}

# --- Helper Functions ---

def submit_chunk_to_mathpix(temp_pdf_path, chunk_identifier):
    """
    Submits a PDF chunk (as a temporary file) to the Mathpix API.
    Returns the pdf_id if successful, None otherwise.
    """
    print(f"Submitting PDF chunk: {chunk_identifier} (from {temp_pdf_path})")
    try:
        with open(temp_pdf_path, "rb") as f:
            payload = {"options_json": json.dumps(MATHPIX_OPTIONS)}
            files = {"file": f}
            headers = {
                "app_id": APP_ID,
                "app_key": APP_KEY
            }

            response = requests.post(SUBMIT_PDF_URL, headers=headers, files=files, data=payload, timeout=180)
            response.raise_for_status()

            result = response.json()
            if "pdf_id" in result:
                print(f"Chunk submitted successfully. PDF ID for {chunk_identifier}: {result['pdf_id']}")
                return result["pdf_id"]
            elif "error_info" in result:
                print(f"Error submitting chunk {chunk_identifier}: {result['error_info']['message']}")
                return None
            else:
                print(f"Unknown error submitting chunk {chunk_identifier}: {result}")
                return None

    except requests.exceptions.RequestException as e:
        print(f"Request failed for chunk {chunk_identifier}: {e}")
        return None
    except FileNotFoundError:
        print(f"Error: Temporary chunk file not found at {temp_pdf_path}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred during submission of {chunk_identifier}: {e}")
        return None

def get_latex_from_mathpix_result(pdf_id, chunk_identifier):
    """
    Retrieves the processed .tex.zip result from Mathpix, extracts .tex content.
    Returns the LaTeX string if successful, None otherwise.
    """
    result_url = f"https://api.mathpix.com/v3/pdf/{pdf_id}.tex.zip"
    headers = {
        "app_id": APP_ID,
        "app_key": APP_KEY
    }
    print(f"Attempting to retrieve LaTeX for chunk '{chunk_identifier}' (PDF ID: {pdf_id})")

    max_retries = 20
    retry_delay = 5  # seconds

    for attempt in range(max_retries):
        try:
            response = requests.get(result_url, headers=headers, timeout=120) # Increased timeout for potentially larger .tex files
            if response.status_code == 200:
                content_type = response.headers.get('Content-Type', '')
                if "application/zip" in content_type:
                    try:
                        with zipfile.ZipFile(BytesIO(response.content)) as zf:
                            # Find the .tex file in the zip (usually one, or named after pdf_id)
                            tex_file_name = None
                            for name in zf.namelist():
                                if name.endswith(".tex"):
                                    tex_file_name = name
                                    break
                            if tex_file_name:
                                with zf.open(tex_file_name) as tex_file:
                                    latex_content = tex_file.read().decode('utf-8')
                                print(f"Successfully extracted LaTeX for chunk {chunk_identifier} from {tex_file_name}.")
                                return latex_content
                            else:
                                print(f"Error: No .tex file found in zip for {chunk_identifier} (PDF ID: {pdf_id}).")
                                return None
                    except zipfile.BadZipFile:
                        print(f"Error: Bad zip file received for {chunk_identifier} (PDF ID: {pdf_id}).")
                        return None
                    except Exception as e_zip:
                        print(f"Error processing zip file for {chunk_identifier} (PDF ID: {pdf_id}): {e_zip}")
                        return None
                else:
                    print(f"Unexpected content type '{content_type}' for {chunk_identifier}. Expected zip.")
                    return None # Or try to decode as text if Mathpix changes behavior?

            elif response.status_code in [404, 202]: # Not found or still processing
                status_message = "Result not yet available (404)" if response.status_code == 404 else "Processing not yet complete (202)"
                print(f"{status_message} for {chunk_identifier}. Retrying in {retry_delay}s... (Attempt {attempt + 1}/{max_retries})")
            else:
                print(f"Error retrieving result for {chunk_identifier}: Status {response.status_code} - {response.text}")
                return None # Permanent error for this attempt

            time.sleep(retry_delay)

        except requests.exceptions.RequestException as e:
            print(f"Request failed while retrieving LaTeX for {chunk_identifier}: {e}")
            if attempt < max_retries - 1:
                print(f"Retrying in {retry_delay}s...")
                time.sleep(retry_delay)
            else:
                print(f"Max retries reached for {chunk_identifier}.")
                return None
    print(f"Failed to retrieve LaTeX for {chunk_identifier} (PDF ID: {pdf_id}) after {max_retries} retries.")
    return None


def process_single_pdf_with_mathpix(pdf_path, all_extracted_latex_data):
    """
    Splits a single PDF into chunks, processes each chunk via Mathpix for LaTeX,
    and appends extracted data to the main list.
    """
    pdf_filename_base = os.path.splitext(os.path.basename(pdf_path))[0]
    print(f"\n--- Processing PDF for LaTeX: {pdf_filename_base}.pdf ---")
    temp_dir_for_chunks = os.path.join(OUTPUT_DIRECTORY, "temp_chunks")
    os.makedirs(temp_dir_for_chunks, exist_ok=True)

    try:
        with open(pdf_path, 'rb') as f_pdf:
            reader = PdfReader(f_pdf)
            total_pages = len(reader.pages)
            pages_per_chunk = 10 # Mathpix might handle larger files well, but chunking for consistency

            for i, start_page_idx in enumerate(range(0, total_pages, pages_per_chunk)):
                end_page_idx = min(start_page_idx + pages_per_chunk, total_pages)
                chunk_identifier = f"{pdf_filename_base}_chunk{i+1}_p{start_page_idx+1}-{end_page_idx}"
                temp_chunk_pdf_path = os.path.join(temp_dir_for_chunks, f"{chunk_identifier}.pdf")
                
                print(f"  Creating chunk: {chunk_identifier} ({start_page_idx+1}-{end_page_idx})")
                writer = PdfWriter()
                for page_num in range(start_page_idx, end_page_idx):
                    writer.add_page(reader.pages[page_num])
                
                with open(temp_chunk_pdf_path, "wb") as temp_chunk_file:
                    writer.write(temp_chunk_file)

                if os.path.getsize(temp_chunk_pdf_path) == 0:
                    print(f"  Skipping empty chunk file: {chunk_identifier}")
                    os.remove(temp_chunk_pdf_path)
                    continue

                mathpix_pdf_id = submit_chunk_to_mathpix(temp_chunk_pdf_path, chunk_identifier)
                
                if mathpix_pdf_id:
                    latex_content = get_latex_from_mathpix_result(mathpix_pdf_id, chunk_identifier)
                    if latex_content:
                        all_extracted_latex_data.append({
                            "source_pdf": pdf_filename_base,
                            "chunk_id": chunk_identifier,
                            "latex_content": latex_content
                        })
                        print(f"  Successfully processed chunk {chunk_identifier}, {len(latex_content)} chars of LaTeX.")
                    else:
                        print(f"  Failed to get LaTeX for chunk {chunk_identifier}.")
                else:
                    print(f"  Failed to submit chunk {chunk_identifier} to Mathpix.")
                
                # Clean up temporary chunk PDF
                try:
                    os.remove(temp_chunk_pdf_path)
                except OSError as e:
                    print(f"  Warning: Could not delete temp chunk {temp_chunk_pdf_path}: {e}")
                
                time.sleep(5) # Be respectful to the API between chunk submissions

                # Save checkpoint every 10 chunks
                if (i + 1) % 10 == 0:
                    checkpoint_csv_path = os.path.join(OUTPUT_DIRECTORY, f"checkpoint_{pdf_filename_base}_chunk{i+1}.csv")
                    try:
                        df = pd.DataFrame(all_extracted_latex_data)
                        # Ensure correct column order
                        cols = ["source_pdf", "chunk_id", "latex_content"]
                        df = df[cols]
                        df.to_csv(checkpoint_csv_path, index=False, encoding='utf-8')
                        print(f"Checkpoint saved to {checkpoint_csv_path} after processing {i+1} chunks")
                    except Exception as e:
                        raise e

    except Exception as e:
        print(f"Error processing PDF file {pdf_path}: {e}")
        import traceback
        traceback.print_exc()

    # Clean up temp_chunks directory if it's empty, otherwise leave for inspection on error
    try:
        if not os.listdir(temp_dir_for_chunks):
            os.rmdir(temp_dir_for_chunks)
    except OSError:
        pass # Ignore if not empty or other issues

    # Print first 500 characters of the plain text from the LaTeX content
    pdf_latex_content = ""
    for item in all_extracted_latex_data:
        if item["source_pdf"] == pdf_filename_base:
            pdf_latex_content += item["latex_content"]
    
    if pdf_latex_content:
        print(pdf_latex_content)
    else:
        print(f"\nNo LaTeX content extracted from {pdf_filename_base}.pdf")

    print(f"--- Finished processing PDF: {pdf_filename_base}.pdf ---")


def process_all_pdfs_for_latex():
    """
    Processes all PDF files in PDF_DIRECTORY using Mathpix for LaTeX extraction.
    """
    if APP_ID == "YOUR_APP_ID" or APP_KEY == "YOUR_APP_KEY":
        print("ERROR: Please replace 'YOUR_APP_ID' and 'YOUR_APP_KEY' with your actual Mathpix credentials at the top of the script.")
        return

    if not os.path.exists(PDF_DIRECTORY):
        print(f"Error: PDF directory '{PDF_DIRECTORY}' does not exist.")
        return

    if not os.path.exists(OUTPUT_DIRECTORY):
        os.makedirs(OUTPUT_DIRECTORY)
        print(f"Created output directory: '{OUTPUT_DIRECTORY}'")

    pdf_files = glob.glob(os.path.join(PDF_DIRECTORY, "*.pdf"))
    if not pdf_files:
        print(f"No PDF files found in '{PDF_DIRECTORY}'.")
        return

    print(f"Found {len(pdf_files)} PDF(s) to process in '{PDF_DIRECTORY}' for LaTeX extraction.")
    
    all_extracted_latex_data = []

    for pdf_path in pdf_files:
        process_single_pdf_with_mathpix(pdf_path, all_extracted_latex_data)
        print(f"Completed processing for {os.path.basename(pdf_path)}. Total LaTeX items collected: {len(all_extracted_latex_data)}")
        time.sleep(10) # Wait a bit longer between processing entire PDF files

    # --- Save Results ---
    if all_extracted_latex_data:
        print(f"\nExtraction finished. Collected {len(all_extracted_latex_data)} LaTeX content blocks.")
        
        try:
            df = pd.DataFrame(all_extracted_latex_data)
            # Ensure correct column order
            cols = ["source_pdf", "chunk_id", "latex_content"]
            df = df[cols]

            output_csv_path = os.path.join(OUTPUT_DIRECTORY, OUTPUT_CSV_FILENAME)
            df.to_csv(output_csv_path, index=False, encoding='utf-8')
            print(f"LaTeX results saved to {output_csv_path}")

        except Exception as e:
            print(f"Error creating DataFrame or saving results to CSV: {str(e)}")
            raw_json_path = os.path.join(OUTPUT_DIRECTORY, "extracted_latex_raw.json")
            try:
                with open(raw_json_path, 'w', encoding='utf-8') as f_json:
                    json.dump(all_extracted_latex_data, f_json, indent=2)
                print(f"Raw extracted LaTeX data saved to {raw_json_path} for debugging.")
            except Exception as json_err:
                print(f"Could not save raw data to JSON: {json_err}")
    else:
        print("\nNo LaTeX data extracted from any PDFs.")
        output_csv_path = os.path.join(OUTPUT_DIRECTORY, OUTPUT_CSV_FILENAME)
        try:
            pd.DataFrame(columns=["source_pdf", "chunk_id", "latex_content"]).to_csv(output_csv_path, index=False, encoding='utf-8')
            print(f"Empty CSV created at {output_csv_path}")
        except Exception as e:
            print(f"Error creating empty CSV file: {str(e)}")

# --- Main Execution ---
if __name__ == "__main__":
    process_all_pdfs_for_latex()
    print("\nScript finished.")