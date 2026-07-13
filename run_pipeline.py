import sys
from src.pipeline import run_pipeline
from src.utils.helpers import setup_logger

logger = setup_logger("runner")

def main():
    """
    Main runner script to execute the POWERGRID data collection and preprocessing pipeline.
    Usage: python run_pipeline.py [onehot|ordinal]
    Default encoding method: ordinal
    """
    method = "ordinal"
    if len(sys.argv) > 1:
        arg = sys.argv[1].lower()
        if arg in ["onehot", "ordinal"]:
            method = arg
        else:
            logger.warning(f"Invalid encoding method '{arg}'. Falling back to 'ordinal'.")
            
    logger.info(f"Starting pipeline execution with categorical encoding method: '{method}'")
    
    try:
        shapes = run_pipeline(method=method)
        logger.info("Pipeline executed successfully!")
        print("\n" + "="*50)
        print("POWERGRID DATA PREPROCESSING SUMMARY")
        print("="*50)
        print(f"Raw Dataset Shape:           {shapes['raw_shape']}")
        print(f"Train Dataset Shape (70%):   {shapes['train_shape']}")
        print(f"Val Dataset Shape (15%):     {shapes['val_shape']}")
        print(f"Test Dataset Shape (15%):    {shapes['test_shape']}")
        print(f"Combined Processed Shape:    {shapes['processed_all_shape']}")
        print("-"*50)
        print("Generated Outputs:")
        print("1. Processed Dataset:       data/processed/processed_dataset.csv")
        print("2. Raw/Clean Validation:    reports/validation_report.csv")
        print("3. Feature Summary Statistics: reports/feature_summary.csv")
        print("4. Serialized ML Encoders:   models/")
        print("="*50 + "\n")
        
    except Exception as e:
        logger.error(f"Pipeline execution failed with error: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
