# Plagiarism Detection Application

This project is a plagiarism detection application built using Flask. It allows users to upload PDF files and check for plagiarism by comparing the extracted text with search engine results.

## Project Structure

```
plagiarism_detection_app
├── src
│   ├── api_flask.py          # Entry point for the Flask application, defines API routes
│   ├── utils
│   │   ├── __init__.py       # Initializes the utils module
│   │   ├── pdf_utils.py       # Contains functions for extracting text from PDF files
│   │   └── search_utils.py    # Contains functions for interacting with search APIs
├── requirements.txt           # Lists the dependencies required for the project
└── README.md                  # Documentation for the project
```

## Installation

To set up the project, follow these steps:

1. Clone the repository:
   ```
   git clone <repository-url>
   cd plagiarism_detection_app
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

1. Start the Flask application:
   ```
   python src/api_flask.py
   ```

2. Use a tool like Postman or cURL to send a POST request to the `/check-plagiarism` endpoint with the following parameters:
   - `pdf_file`: The PDF file to check for plagiarism.
   - `serper_api_key`: Your API key for the search service.
   - `query`: (Optional) A query string to refine the search.

## API Endpoints

- **GET /**: A simple endpoint to verify that the API is running.
- **POST /check-plagiarism**: Checks the uploaded PDF file for plagiarism.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.

## License

This project is licensed under the MIT License. See the LICENSE file for details.