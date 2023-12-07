import requests
import matplotlib.pyplot as plt

res = requests.get('http://localhost:3006/auth/login')

print(res)

if res :
    print("ok")
else:
    print("erreur")

print(res.status_code)
print(res.headers)