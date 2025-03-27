# LeetLingo Backend

LeetLingo Backend is a FastAPI-based service that provides speech analysis and interview question generation capabilities. The service analyzes spoken language for various parameters including accent, clarity, confidence, and vocabulary.

## Features

- Speech Analysis
  - Accent detection and scoring
  - Clarity and articulation assessment
  - Confidence and tone evaluation
  - Vocabulary and language use analysis
  - Overall speech quality scoring

- Interview Question Generation
  - Company-specific questions
  - Role-based question generation
  - AWS Bedrock integration for knowledge base queries

## Prerequisites

- Python 3.11.x
- OpenAI API Key
- AWS Credentials (for Bedrock integration)
- Required Python packages (see requirements.txt)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/ayam04/leetlingo-backend.git
cd leetlingo-backend
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Create a `.env` file with the following environment variables:
```env
OPENAI_API_KEY=your_openai_api_key
region=your_aws_region
aws_access_key_id=your_aws_access_key
aws_secret_access_key=your_aws_secret_key
```

## API Endpoints

### 1. Score Language (/score)
- **Method**: POST
- **Input**: Audio file (WAV format)
- **Output**: Comprehensive speech analysis including:
  - Overall score
  - Accent analysis
  - Clarity score
  - Confidence evaluation
  - Vocabulary assessment
  - Detailed feedback summary

### 2. Generate Questions (/get_questions)
- **Method**: POST
- **Input**: JSON with company and role information
- **Parameters**:
  - company: Company name
  - role: Job role

## Technical Details

- Built with FastAPI for high performance
- Uses OpenAI's GPT-4 for speech analysis
- Integrates with AWS Bedrock for question generation
- Implements concurrent processing for faster analysis
- Supports CORS for cross-origin requests
- Uses librosa for audio processing

## Running the Application

Start the server with:
```bash
python app.py
```
The server will run on `http://localhost:8080` with auto-reload enabled.

## Dependencies

Key dependencies include:
- fastapi
- librosa==0.10.2.post1
- openai==1.35.14
- boto3==1.35.10
- pydantic==2.9.2
- uvicorn
- python-dotenv
- numpy
- soundfile
- httpx==0.27.2

## Error Handling

The application includes comprehensive error handling for:
- Audio file processing
- API interactions
- Speech analysis
- AWS service integration

## Notes

- Maximum audio length is set to 10 seconds
- Audio is processed at 44kHz sample rate
- Results are automatically saved to `results.json` for each question
- All API responses are in JSON format