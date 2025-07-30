import argparse
import joblib
import warnings
import pandas as pd

warnings.filterwarnings('ignore')


def main(args):
    try:
        # Load model
        pkl = joblib.load(args.model_file)

        # decompose pre processor and model
        model = pkl['model']
        preprocessor = pkl['preprocessor']

        # Load input data
        test = pd.read_parquet(args.input_file).drop(columns=['default'], errors='ignore')

        # Transform features
        X_test = preprocessor.transform(test)

        ## Predict probabilities
        probs = model.predict(X_test)

        # Convert probabilities to 0/1 using 0.5 threshold
        predictions = (probs > 0.5).astype(int)

        # check if prediction length matches input
        assert len(predictions) == len(
            test), f"Prediction length {len(elasticnet_pred)} does not match input length {len(test)}"

        # saving the sample predictions
        print('saving sample predictions')
        output_path = args.output_file

        # Create a DataFrame for submission
        submission_df = pd.DataFrame({'default': predictions.flatten()})
        submission_df.to_parquet(output_path, index=False)

        print('Predictions saved to:', output_path)
    except Exception as e:
        print(f'Exception raised: {e}')
        return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_file", type=str, required=True)
    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--output_file", type=str, required=True)
    args = parser.parse_args()
    main(args)
