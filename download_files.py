import requests

# URL of the image
image_url = "https://img.freepik.com/free-vector/isolated-tree-white-background_1308-26130.jpg?w=2000"
logo_url= "https://www.pngegg.com/en/png-whkxy/download"
# Send a HTTP GET request to the image URL
response = requests.get(logo_url)

# Check if the request was successful
if response.status_code == 200:
    # Write the contents of the response (i.e., the image) to a file
    with open("base_image_sampe.jpg", "wb") as f:
        f.write(response.content)
else:
    print(f"Failed to download the image. HTTP status code: {response.status_code}")

