import requests
from bs4 import BeautifulSoup

req = requests.get('http://www.weather.go.kr/weather/climate/past_cal.jsp?stn=108&yy=2019&mm=3&x=8&y=7&obs=1')

html = req.text
header = req.headers
status = req.status_code
is_ok = req.ok

soup = BeautifulSoup(html, 'html.parser')

print(soup)
