#!/usr/bin/env python3
"""
Resume Status Checker for LBAD Project

This utility script checks the status of resumeable processing for both 
logbert_embeddings.py and transformer.py, showing what's completed and 
what can be resumed.
"""

import sys
from pathlib import Path

# Add src directory to path
sys.path.append(str(Path(__file__).parent / "src"))

from logbert_embeddings import (
    find_available_log_types, check_existing_outputs, 
    CHECKPOINT_DIR as LOGBERT_CHECKPOINT_DIR, OUTPUT_DIR
)
from transformer import (
    find_available_embeddings, analyze_embedding_types, 
    check_existing_results, detect_system_resources,
    CHECKPOINT_DIR as TRANSFORMER_CHECKPOINT_DIR, RESULTS_DIR, MODELS_DIR
)

def check_logbert_status():
    """Check LogBERT embeddings processing status"""
    print("🔬 LogBERT Embeddings Status")
    print("=" * 50)
    
    available_types = find_available_log_types()
    if not available_types:
        print("❌ No processed log types found")
        return
    
    for log_type in available_types:
        status = check_existing_outputs(log_type)
        
        # Check for checkpoints
        checkpoint_pattern = f"{log_type}_embeddings_*.pkl"
        checkpoints = list(LOGBERT_CHECKPOINT_DIR.glob(checkpoint_pattern)) if LOGBERT_CHECKPOINT_DIR.exists() else []
        
        print(f"\n📁 {log_type}:")
        print(f"  Log embeddings: {'✅' if status['log_embeddings'] else '❌'}")
        print(f"  Label vectors:  {'✅' if status['label_vectors'] else '❌'}")
        print(f"  Attack types:   {'✅' if status['attack_types'] else '❌'}")
        print(f"  Visualization:  {'✅' if status['visualization'] else '❌'}")
        print(f"  Checkpoints:    {len(checkpoints)} found")
        
        if status['complete']:
            print(f"  Status: ✅ Complete")
        elif any(status.values()):
            print(f"  Status: 🔄 Partial (can resume)")
        else:
            print(f"  Status: ⏳ Not started")
    
    # Check combined status
    combined_status = check_existing_outputs("all_combined")
    combined_checkpoints = list(LOGBERT_CHECKPOINT_DIR.glob("all_combined_embeddings_*.pkl")) if LOGBERT_CHECKPOINT_DIR.exists() else []
    
    print(f"\n📁 all_combined:")
    print(f"  Status: {'✅ Complete' if combined_status['complete'] else '🔄 Partial' if any(combined_status.values()) else '⏳ Not started'}")
    print(f"  Checkpoints: {len(combined_checkpoints)} found")

def check_transformer_status():
    """Check Transformer training status"""
    print("\n🤖 Transformer Training Status")
    print("=" * 50)
    
    # Detect system for proper config
    config = detect_system_resources()
    
    available_types = find_available_embeddings()
    if not available_types:
        print("❌ No embedding files found")
        return
    
    # Analyze embedding types
    embedding_analysis = analyze_embedding_types(available_types)
    
    for log_type in available_types:
        status = check_existing_results(log_type, config)
        
        # Check for training checkpoints
        checkpoint_pattern = f"{log_type}_epoch_*.pth"
        checkpoints = list(TRANSFORMER_CHECKPOINT_DIR.glob(checkpoint_pattern)) if TRANSFORMER_CHECKPOINT_DIR.exists() else []
        
        # Get embedding info
        embed_info = embedding_analysis.get(log_type, {})
        embed_type = embed_info.get('embedding_type', 'Unknown')
        embed_dim = embed_info.get('dimension', 0)
        
        print(f"\n📁 {log_type} ({embed_type} - {embed_dim}D):")
        print(f"  Results file:   {'✅' if status['results_pkl'] else '❌'}")
        print(f"  Labels file:    {'✅' if status['labels_pkl'] else '❌'}")
        print(f"  Report:         {'✅' if status['classification_report'] else '❌'}")
        print(f"  Visualizations: {'✅' if status['visualizations'] else '❌'}")
        print(f"  Model saved:    {'✅' if status['model_saved'] else '❌'}")
        print(f"  Train checkpoints: {len(checkpoints)} found")
        
        if status['complete']:
            print(f"  Status: ✅ Complete")
        elif any(status.values()):
            print(f"  Status: 🔄 Partial (can resume)")
        else:
            print(f"  Status: ⏳ Not started")

def show_resume_commands():
    """Show useful resume commands"""
    print("\n💡 Resume Commands")
    print("=" * 50)
    
    print("📋 LogBERT Embeddings:")
    print("  Resume all:           python src/logbert_embeddings.py")
    print("  Resume specific:      python src/logbert_embeddings.py --log-type wp-access")
    print("  Force restart:        python src/logbert_embeddings.py --force-restart")
    print("  Clean checkpoints:    python src/logbert_embeddings.py --clean-checkpoints")
    
    print("\n🤖 Transformer Training:")
    print("  Resume all:           python src/transformer.py")
    print("  Resume specific:      python src/transformer.py --log-type wp-access")
    print("  Force restart:        python src/transformer.py --force-restart")
    print("  Clean checkpoints:    python src/transformer.py --clean-checkpoints")
    
    print("\n🗂️  Checkpoint Locations:")
    print(f"  LogBERT:     {LOGBERT_CHECKPOINT_DIR}/")
    print(f"  Transformer: {TRANSFORMER_CHECKPOINT_DIR}/")
    
    print("\n📁 Output Locations:")
    print(f"  Embeddings:  {OUTPUT_DIR}/")
    print(f"  Results:     {RESULTS_DIR}/")
    print(f"  Models:      {MODELS_DIR}/")

def show_disk_usage():
    """Show disk usage of various directories"""
    print("\n💾 Disk Usage")
    print("=" * 50)
    
    def get_dir_size(path):
        if not path.exists():
            return 0
        total = 0
        for item in path.rglob('*'):
            if item.is_file():
                total += item.stat().st_size
        return total / (1024**2)  # MB
    
    embeddings_size = get_dir_size(OUTPUT_DIR)
    results_size = get_dir_size(RESULTS_DIR)
    models_size = get_dir_size(MODELS_DIR)
    logbert_checkpoints_size = get_dir_size(LOGBERT_CHECKPOINT_DIR)
    transformer_checkpoints_size = get_dir_size(TRANSFORMER_CHECKPOINT_DIR)
    
    print(f"  Embeddings:              {embeddings_size:.1f} MB")
    print(f"  Results:                 {results_size:.1f} MB")
    print(f"  Models:                  {models_size:.1f} MB")
    print(f"  LogBERT Checkpoints:     {logbert_checkpoints_size:.1f} MB")
    print(f"  Transformer Checkpoints: {transformer_checkpoints_size:.1f} MB")
    print(f"  Total:                   {sum([embeddings_size, results_size, models_size, logbert_checkpoints_size, transformer_checkpoints_size]):.1f} MB")

def main():
    """Main status checker"""
    print("🔍 LBAD Resumeable Processing Status Checker")
    print("=" * 60)
    
    try:
        check_logbert_status()
        check_transformer_status()
        show_resume_commands()
        show_disk_usage()
        
        print("\n✅ Status check complete!")
        
    except Exception as e:
        print(f"❌ Error checking status: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 