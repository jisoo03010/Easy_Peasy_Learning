# 설치

1. 파이선 라이브러리 설치
```bash
$ pip install -r requirements.txt
``` 

# 실행

### 1. flask 서버 기반의 RestAPI 

- ##### http 기반 서버 실행 명령어
```
$ python model_interface.py --port 8080

```

* API 명세

<table><tbody>
<tr>
  <th>API Name</th>
  <th>API URL</th>
  <th>Method</th>
  <th>Parameters</th>
<tr>
<tr>
  <td>Binary Classification</td>
  <td>/inference/binary_classification</td>
  <td>POST</td>
  <td>

***HEADER***
  * Content-Type: multipart/form-data

***BODY***
  * image : [image blob]
  </td>
<tbody></table>

  

* API 응답 결과 - 사람이 있는 경우
```json

```

# 이미지 이진분류 모델 학습 및 테스트 

### 1. 모델 학습 실행
```bash
$ python mobilenet_img_model.py [옵션]
``` 
### 2. 모델 테스트 실행
```bash
$ python mobilenet_img_model.py --test [옵션]
``` 
* 모델 실행 명령어 옵션 값

<table><tbody>
<tr>
  <th>Option Value</th>
  <th>Default Value</th>
  <th>Type</th>
</tr>
<tr> 
  <td>--epoch</td>
  <td>60</td>
  <td>int</td>
</tr>

<tr> 
  <td>--early-stopping-epochs</td>
  <td>5</td>
  <td>int</td>
</tr>

<tr> 
  <td>--batch-size</td>
  <td>64</td>
  <td>int</td>
</tr>

<tr> 
  <td>--lr</td>
  <td>0.001</td>
  <td>float</td>
</tr>

<tr> 
  <td>--datasets-path</td>
  <td>"./sample_data"</td>
  <td>str</td>
</tr>
<tr> 
  <td>--model-save-path</td>
  <td>"./mobilenet_v{timestamp}.pt"</td>
  <td>str</td>
</tr>

<tr> 
  <td>--model-file-path</td>
  <td>"./mobilenet_v6.pt"</td>
  <td>str</td>
</tr>
<tr> 
  <td>--test</td>
  <td></td>
  <td></td>
</tr>
<tbody></table>

