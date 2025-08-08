import requests

def ping_app(app_url):
    try:
        response = requests.get(app_url)
        if response.status_code == 200:
            print(f"Successfully pinged {app_url}.")
        else:
            print(f"Failed to ping {app_url} with status code: {response.status_code}.")
    except requests.exceptions.RequestException as e:
        print(f"Error pinging {app_url}: {e}.")

# Replace with your app's URL
app_url = "https://predictable.streamlit.app/"
ping_app(app_url)
