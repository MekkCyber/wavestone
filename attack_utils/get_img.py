import requests
import matplotlib.pyplot as plt

res = requests.get('http://localhost:3006/auth/login')

print(res)
