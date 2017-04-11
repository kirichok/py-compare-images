from tineyeservices import MatchEngineRequest, Image
api = MatchEngineRequest(api_url='http://localhost/rest/', username=None, password=None)

image1 = Image(filepath='./images/4.jpg')
image2 = Image(filepath='./images/1.jpg')

api.compare_image(image_1=image1, image_2=image2)

