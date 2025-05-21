# Persona Prediction Application

This project is a full-stack application that predicts a user's digital persona based on their bio and social media posts. It supports two prediction modes: a trained model and a zero-shot model.

## Features

- **Bio and Posts Input**: Users can enter their bio and up to 5 social media posts.
- **Prediction Modes**: Toggle between a trained model and a zero-shot model for predictions.
- **Confidence Display**: Shows the confidence level of the prediction as a percentage.
- **Responsive UI**: Built with React and Chakra UI for a modern user experience.

## Prerequisites

- Python 3.8 or higher
- Node.js and npm (for the frontend)
- Git

## Installation

### Backend

1. Navigate to the backend directory:
   ```sh
   cd backend
   ```

2. Create a virtual environment and activate it:
   ```sh
   python -m venv venv
   source venv/bin/activate  # On Windows, use: venv\Scripts\activate
   ```

3. Install the required packages:
   ```sh
   pip install -r requirements.txt
   ```

### Frontend

1. Navigate to the frontend directory:
   ```sh
   cd frontend-react
   ```

2. Install the required packages:
   ```sh
   npm install
   ```

## Running the Application

### Backend

1. Start the FastAPI server:
   ```sh
   uvicorn app:app --reload
   ```

   The backend will be available at `http://localhost:8000`.

### Frontend

1. Start the React development server:
   ```sh
   npm start
   ```

   The frontend will be available at `http://localhost:3000`.

## Usage

1. Open your browser and go to `http://localhost:3000`.
2. Enter your bio and at least 3 social media posts.
3. Toggle between the trained model and zero-shot model using the switch.
4. Click "Predict My Persona" to see the results.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
