import sys
import os
import json
import time
from pathlib import Path

# Add paths to find local libs
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from cjm_text_plugin_nltk.plugin import NLTKPlugin

def title(msg):
    print(f"\n{'='*60}\n{msg}\n{'='*60}")

def run_nltk_test():
    title("TEST: NLTK Text Plugin (Sentence Splitting & Span Tracking)")

    # 1. Setup
    raw_text = "Laying Plans Sun Tzu said, The art of war is of vital importance to the state. It is a matter of life and death, a road either to safety or to ruin. Hence it is a subject of inquiry which can on no account be neglected."
    
    # 2. Initialize Plugin
    print("Initializing Plugin...")
    plugin = NLTKPlugin()
    
    # This will trigger the NLTK download logic
    plugin.initialize({
        "language": "english"
    })

    # 3. Execute
    print("\n--- Processing Text ---")
    start_time = time.time()
    
    # execute() returns a dict (simulating IPC)
    result_dict = plugin.execute(action="split_sentences", text=raw_text)
    
    duration = time.time() - start_time
    print(f"Processing took {duration:.4f}s")
    
    # 4. Validation
    spans = result_dict['spans']
    print(f"Detected {len(spans)} sentences.")
    
    print("\n--- Verifying Spans ---")
    for i, span in enumerate(spans):
        # Reconstruct text using indices to prove accuracy
        original_slice = raw_text[span['start_char']:span['end_char']]
        
        print(f"[{i}] {span['start_char']}->{span['end_char']}: {span['text'][:30]}...")
        
        assert span['text'] == original_slice, f"Mismatch at index {i}: '{span['text']}' vs '{original_slice}'"
        assert span['label'] == "sentence"

    # 5. Cleanup
    plugin.cleanup()
    print("\n[SUCCESS] NLTK Plugin verified.")

if __name__ == "__main__":
    try:
        run_nltk_test()
    except Exception as e:
        print(f"\n!!! FAILED !!!\n{e}")
        import traceback
        traceback.print_exc()