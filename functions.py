import openai
from dotenv import load_dotenv
import os
import boto3
import json
from botocore.exceptions import ClientError

load_dotenv()

KNOWLEDGE_BASE_ID = "EY0ZGLB9OT"
MODEL_ARN = "arn:aws:bedrock:us-east-1::foundation-model/amazon.titan-text-premier-v1:0"

openai_api_key = os.getenv("OPENAI_API_KEY")
openai.api_key = openai_api_key

bedrock = boto3.client('bedrock-runtime', 
    region_name=os.getenv('region'),
    aws_access_key_id=os.getenv('aws_access_key_id'),
    aws_secret_access_key=os.getenv('aws_secret_access_key')
)

def get_summary(category, score, transcription):
    prompt = f"Analyze the following transcription based on the parameters of {category}. Provide a score from 1 to 10, and a professional summary explaining the reasoning behind the score. Include areas of strength and specific suggestions for improvement where applicable: '{transcription}' with a score of {score}/10. Answer in a FEW LINES, NOT IN POINTS. Talk from a 3rd person perspective, in respect to the candidate. DO NOT SHOW OR TALK ABOUT the score in the summary, just talk about the candidate performance WITHOUT SPECIFYING THE SCORE."
    
    try:
        client = openai.OpenAI()
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that provides detailed but concise speech analysis based on various parameters. You analyse candidate interview."},
                {"role": "user", "content": prompt}
            ],
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Failed to generate summary for {category}: {str(e)}"

def get_overall_summary(data):
    prompt = f"Provide an Overall summary detailing the speaker's strengths, opportunities for improvement, and specific steps to enhance the overall communication style. Here the Overall Candidate summary: {data}. Answer in a FEW LINES, NOT IN POINTS. Talk from a 3rd person perspective, in respect to the candidate. DO NOT SHOW OR TALK ABOUT the score in the summary, just talk about the candidate performance WITHOUT SPECIFYING THE SCORE."

    try:
        client = openai.OpenAI()
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that provides detailed but concise speech analysis based on various parameters. You analyse candidate interview."},
                {"role": "user", "content": prompt}
            ],
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Failed to generate overall summary: {str(e)}"

def query_bedrock_knowledge_base(query: str):
    try:
        response = bedrock.invoke_model(
            body=json.dumps({
                'inputText': query
            }),
            modelId=MODEL_ARN,
            contentType='application/json',
            accept='application/json',
            trace='DISABLED'
        )
        
        response_body = response['body'].read()
        response_data = json.loads(response_body)
        
        if 'results' in response_data and len(response_data['results']) > 0:
            output_text = response_data['results'][0].get('outputText')
            if output_text:
                return output_text
            else:
                raise ValueError('No text in the response')
        else:
            raise ValueError('No results found in the response')
    except ClientError as e:
        print(f"Error invoking Bedrock knowledge base: {e}")
        raise

def get_aws_questions(company: str, role: str):
    prompt = f"""Generate 5 interview questions for a {role} position at {company}. 
    The questions should be a mix of functional, situational, and behavioral questions 
    that are relevant to the role and company. ONLY GIVE ME 5 QUESTIONS WITHOUT THE ANSWERS.

    Format your response as a JSON object with the following structure:
    {{"questions": ["question1", "question2", "question3", "question4", "question5"]}}
    """
    
    try:
        response = query_bedrock_knowledge_base(prompt)
        questions_json = json.loads(response)
        
        questions = questions_json.get('questions', [])[:5]
        
        while len(questions) < 5:
            questions.append(f"Additional question for {role} at {company}")
        
        return {"questions": questions}
    except Exception as e:
        print(f"Error querying Bedrock: {str(e)}")
        return {"error": str(e)}