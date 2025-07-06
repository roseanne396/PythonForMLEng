# Import required libraries
import sys
import os
import datetime
from sentence_transformers import SentenceTransformer
import joblib


# Check that the user gave 2 arguments: input file and source name
def main():

    if len(sys.argv) != 3:
        print("Usage: python score_headlines.py <input_file.txt> <source_name>")
        sys.exit(1)

    input_file = sys.argv[1]
    source_name = sys.argv[2]

    # Read the input file, making sure it exists and is not empty
    if not os.path.isfile(input_file):
        print(f"Error: The file '{input_file}' does not exist.")
        sys.exit(1)

    with open(input_file, "r", encoding="utf-8") as f:
        headlines = [line.strip() for line in f if line.strip()]

    if not headlines:
        print("Error: No headlines found in the input file.")
        sys.exit(1)

    # Load the SentenceTransformer Model
    st_model = SentenceTransformer("/opt/huggingface_models/all-MiniLM-L6-v2")

    # Convert Headlines to Embeddings
    embeddings = st_model.encode(headlines)

    # Load the Pretrained SVM Model
    svm_model = joblib.load("model/svm_model.pkl")

    # Making predictions
    predictions = svm_model.predict(embeddings)

    # Create Output Filename
    today = datetime.date.today().strftime("%Y_%m_%d")
    output_filename = f"headline_scores_{source_name}_{today}.txt"

    # Save Predictions to File
    with open(output_filename, "w", encoding="utf-8") as f:
        for label, headline in zip(predictions, headlines):
            f.write(f"{label},{headline}\n")

    print(f"Results saved to {output_filename}")


if __name__ == "__main__":
    main()
