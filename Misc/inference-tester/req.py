import requests
import json

# Specify the URL of your Flask API
url = 'http://192.168.16.158:30001/inference'

# Specify the path to the image file
image_path = '000000000139.jpg'

# Create the payload with the image file
payload = {'image': open(image_path, 'rb')}

# Send the POST request
response = requests.post(url, files=payload)

# Check if the request was successful
if response.status_code == 200:
    # Process the JSON response
    result = response.json()
    print(result)

    # Extract and print the desired fields
#    scores = result['scores']
#    labels = result['labels']
#    print('Scores:', scores)
#    print('Labels:', labels)
else:
    # Print the error message if the request failed
    print('Error:', response.text)
