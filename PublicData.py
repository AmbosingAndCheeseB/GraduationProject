from urllib.request import urlopen, Request
from urllib.parse import urlencode, quote_plus

url = 'http://apis.data.go.kr/B090041/openapi/service/SpcdeInfoService/getSundryDayInfo'
queryParams = '?' + urlencode({quote_plus('ServiceKey'): 'Kpd29qGQu%2BwnyUKVX5jBenSRbn1vw8Fz9HCVc%2FqKDa67lvi9RX0YD0YmR20coGHc3o08DtM03EmGtHUHzaPMKw%3D%3D',
                               quote_plus('solYear'): '2015', quote_plus('solMonth'): '10'})

request = Request(url + queryParams)
request.get_method = lambda: 'GET'
response_body = urlopen(request).read()
print(response_body)
