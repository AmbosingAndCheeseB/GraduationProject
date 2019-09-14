import requests
from bs4 import BeautifulSoup
from openpyxl import Workbook

req = requests.get('http://www.weather.go.kr/weather/climate/past_cal.jsp?stn=108&yy=2019&mm=1&x=14&y=10&obs=1')

if req.ok:
    html = req.text
    soup = BeautifulSoup(html, 'html.parser')

wb = Workbook()
sheet1 = wb.active

weather = soup.select('td.align_left')

i = 1
for x in weather:
    txt = x.text
    sheet1.cell(i, 1, txt[5:9])
    i = i+1

wb.save('./crawler.xlsx')
