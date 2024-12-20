from transformers import pipeline
import time
import logging
from datetime import datetime
import json

# defined a class to analayse the sentiment and initialised list and analyzer with none
class CustomSentimentEngine:
    def __init__(self):
        self.analyzer = None
        self.analysis_history = []
        self.setup_logging()
        self.initialize_engine()
        
    def setup_logging(self):
        logging.basicConfig(
            filename=f'sentiment_analysis_{datetime.now().strftime("%Y%m%d")}.log',
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)

    def initialize_engine(self):
        self.logger.info("Initializing sentiment analysis engine...")
        try:
            self.analyzer = pipeline("sentiment-analysis")
            self.logger.info("Engine initialized successfully")
        except Exception as error:
            self.logger.error(f"Engine initialization failed: {str(error)}")
            raise SystemExit("Critical error: Unable to initialize sentiment analysis engine")

    def analyze_sentiment(self, text_input):
        try:
            if not isinstance(text_input, str):
                return self._create_error_response("Input must be a string")
            
            text = ' '.join(text_input.split()).strip()
            
            if not text:
                return self._create_error_response("Empty text provided")

            start_time = time.time()
            result = self.analyzer(text)
            analysis_time = time.time() - start_time

            enhanced_result = {
                "timestamp": datetime.now().isoformat(),
                "input_text": text,
                "sentiment": {
                    "label": result[0]['label'],
                    "confidence": f"{result[0]['score']*100:.2f}%"
                },
                "analysis_time_ms": f"{analysis_time*1000:.2f}",
                "status": "success"
            }
            
            self.analysis_history.append(enhanced_result)
            return enhanced_result

        except Exception as error:
            self.logger.error(f"Analysis error: {str(error)}")
            return self._create_error_response(str(error))

    def _create_error_response(self, error_message):
        return {
            "timestamp": datetime.now().isoformat(),
            "status": "error",
            "error_message": error_message
        }

def run_analysis():
    print("\n---Sentiment Analysis System ---")
    print("Initializing system...")
    
    analyzer = CustomSentimentEngine()
    
    while True:
        print("\nOptions:")
        print("1. Analyze text")
        print("2. View history")
        print("3. Exit")
        
        choice = input("\nSelect an option (1-3): ")
        
        if choice == '1':
            text = input("\nEnter text to analyze: ")
            result = analyzer.analyze_sentiment(text)
            print("\nAnalysis Result:")
            print(json.dumps(result, indent=2))
            
        elif choice == '2':
            if analyzer.analysis_history:
                print("\nAnalysis History:")
                for item in analyzer.analysis_history:
                    print(json.dumps(item, indent=2))
            else:
                print("\nNo analysis history available.")
                
        elif choice == '3':
            print("\nThank you for using the Sentiment Analysis System!")
            break
            
        else:
            print("\nInvalid option. Please try again.")

if __name__ == "__main__":
    try:
        run_analysis()
    except KeyboardInterrupt:
        print("\nProgram terminated by user.")
    except Exception as e:
        print(f"\nUnexpected error occurred: {str(e)}")