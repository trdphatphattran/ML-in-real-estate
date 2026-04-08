# Real Estate Recommendation System (District 1, 2, 7, Binh Thanh)

## 📌 Tổng quan dự án  
Dự án xây dựng hệ thống gợi ý bất động sản (Căn hộ & Biệt thự) tại các khu vực trọng điểm của TP.HCM bao gồm Quận 1, Quận 2, Quận 7 và Bình Thạnh. Hệ thống sử dụng thuật toán K-Nearest Neighbors (KNN) để tìm kiếm và đề xuất các bất động sản có đặc điểm tương đồng với nhu cầu của người dùng.  

## 🛠 Công nghệ sử dụng
* **Ngôn ngữ:** Python (Flask Framework).  
* **AI/ML:** Scikit-learn (KNN), Google Generative AI (Gemini 2.5 Flash).  
* **Xử lý dữ liệu:** Pandas, NumPy, Regex (Xử lý ngôn ngữ tự nhiên cơ bản).  
* **Giao diện:** HTML/JS & CSS (Giao diện Chatbot tương tác thời gian thực).

## 📊 Dữ liệu
Dữ liệu được xử lý qua pipeline nghiêm ngặt để đảm bảo thuật toán KNN hoạt động chính xác:
* **Chuẩn hóa giá:** Chuyển đổi các định dạng văn bản "tỷ", "triệu" về giá trị số thực để tính toán.
* **Nhận diện địa lý:** Hệ thống sử dụng Regex và Mapping để bắt chính xác các biến thể của tên Quận (ví dụ: "q1", "Q.1", "quan 1" đều về "Quận 1").
* **Xử lý hình ảnh:** Tự động ánh xạ ID bất động sản với kho ảnh local (.jpg, .png, .webp).

## 🤖 Thuật toán KNN trong dự án  
Hệ thống sử dụng kỹ thuật lai giữa lọc dữ liệu (Filtering) và tính toán độ tương đồng:  
* **Tiền xử lý:** Sử dụng **StandardScaler** cho biến số (Giá, Phòng ngủ) và **OneHotEncoder** cho biến phân loại (Quận).
* **Xây dựng vector:** Mỗi căn hộ được biểu diễn bằng một vector đặc trưng trong không gian đa chiều.
* **Tính toán:** Sử dụng metric **Euclidean** để tìm **K** láng giềng gần nhất với yêu cầu của người dùng.
* **AI Integration:** Kết quả từ KNN được đưa vào Prompt làm ngữ cảnh (Context) để Gemini phân tích ưu/nhược điểm từng căn.

## 📈 Kết quả
* **Độ chính xác:** Thuật toán KNN trả về các kết quả có độ tương đồng (Similarity) trên 30% so với nhu cầu.
* **Chatbot thông minh:** Có khả năng phân biệt giữa Smalltalk và tư vấn bất động sản để phản hồi phù hợp.
* **Tư vấn chuyên sâu:** AI không chỉ liệt kê danh sách mà còn biết so sánh giá trị đầu tư, vị trí và tiện ích của từng căn hộ được gợi ý.

## Demo Web  
* **Giả sử nhập:** "Tôi cần tìm mua căn hộ 1 phòng ngủ tại quận 2 giá dưới 15 tỷ".  
<img width="455" height="431" alt="image" src="https://github.com/user-attachments/assets/f403b5bf-4b10-49c1-92c6-9df7cf9eb654" />

## 📂 Cấu trúc thư mục  
```text
├── image                       
├── static
|   └── script.js
|   └── style.css
├── templates
|   ├── index.html
├── app1.py
├── nhadat.csv
└── README.md
```
## 💻 Hướng dẫn sử dụng  
### 1. Clone Repository  
```python
git clone https://github.com/trdphatphattran/Pyspark-lda-news-classification.git](https://github.com/trdphatphattran/ML-in-real-estate.git
cd ML-in-real-estate
```
### 2. Cài thư viện  
```python
pip install -r requirements.txt
```
### 3. Chạy Streamlit  
```python
streamlit run app.py
```
## 📬 Thông tin liên hệ

Nếu bạn có bất kỳ câu hỏi nào về dự án hoặc muốn hợp tác, vui lòng liên hệ với mình qua:

* **Họ và tên:** Trần Đại Phát
* **LinkedIn:** [Phat Tran](https://www.linkedin.com/in/phat-tran-9ba42a341/)
* **GitHub:** [trdphatphattran](https://github.com/trdphatphattran)
* **Email:** phattrandai15062005@gmail.com
* **Phone:** 0908647977 


