import boto3
from decimal import Decimal
import json
import urllib.request
import urllib.parse
import urllib.error
from claude_prompt_template import claude_prompt
from langchain import PromptTemplate

print('Loading function')

# rekognition client
rekognition = boto3.client('rekognition')

# bedrockclient
bedrock_client = boto3.client(
    service_name='bedrock-runtime',
    region_name='us-east-1'
)

# --------------- Prompt elements ------------------ #
# foll prompt needs to be assemebled prefix + labels + suffix
prefix_prompt_claude = prompt_claude
suffix_prompt_claude = "Your task is to create a human readable,  description of the situation. Then decide if the situation you have described is potentially dangerous or not. Please give a short one sentence reasoning for your decision. Assistant:"

# --------------- call Rekognition APIs ------------------

def imageAnalyzer(bucket, key):
    response = rekognition.detect_labels(Image={"S3Object": {"Bucket": bucket, "Name": key}})

    labels = response['Labels']
    print(f'Found {len(labels)} labels in the image:')
    label_names = ''
    for label in labels:
        name = label['Name']
        confidence = label['Confidence']
        #print(f'> Label "{name}" with confidence {confidence:.2f}')
        if confidence>95 and name != 'Person':
            print(name + "|" + str(confidence))
            label_names = label_names + name + ","

    return label_names

# --------------- call to Bedrock/LLM APIs ------------------

def AI_nanny_prompter(prompt,llm_type):
    body = json.dumps({"prompt": prompt,
                    "max_tokens_to_sample":300,
                    "temperature":1,
                    "top_k":250,
                    "top_p":0.999,
                    "stop_sequences":[]})
    modelId = 'anthropic.claude-v2' # change this to use a different version from the model provider
    accept = 'application/json'
    contentType = 'application/json'

    response = bedrock_client.invoke_model(body=body, modelId=modelId, accept=accept, contentType=contentType)
    response_body = json.loads(response.get('body').read())

    response_text_claude = response_body.get('completion')

    return response_text_claude

# --------------- Main handler ------------------


def lambda_function(event, context):
    '''Demonstrates S3 trigger that uses
    Rekognition APIs to detect faces, labels and index faces in S3 Object.
    '''
    # Get the object from the event
    bucket = event['Records'][0]['s3']['bucket']['name']
    key = urllib.parse.unquote_plus(event['Records'][0]['s3']['object']['key'])
    try:
        # image analyzer function
        response = imageAnalyzer(bucket, key)
        # prompt text
        prompt_template_img_descrpition = PromptTemplate.from_template(prompt_claude)
        prompt_for_generation = prompt_template_img_descrpition.format(labels=response)
        response_text = AI_nanny_prompter(prompt_for_generation,"claude")
        print('response_text --- \n' + response_text)
        return response

    except Exception as e:
        print(e)
        print("Error processing object {} from bucket {}. ".format(key, bucket) +
              "Make sure your object and bucket exist and your bucket is in the same region as this function.")
        raise e
