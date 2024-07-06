import requests
 
url = 'https://rcwlogin.azurewebsites.net/login'
 
data = {
    "first_name" : 'Malak',
    'last_name' : 'Baghzali'
}
 
response = requests.post(url, json=data)
 
if response.status_code == 200:
    print(response.json())
else:
    print(f"Failed to connect to the API. Status code: {response.status_code}")