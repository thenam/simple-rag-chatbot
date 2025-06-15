# simple-rag-chatbot

## Embedding model
bkai-foundation-models/vietnamese-bi-encoder

## LLM model
lmsys/vicuna-7b-v1.5

## note
- Chạy project trong môi trường linux hoặc wsl
- Lưu ý phiên bản để ứng dụng có thể chạy tốt và không bị xung đột giữa các thư viện
  + python=3.11
  + torch==2.6.0
  + streamlit==1.36.0

## run:
```commandline
streamlit run main.py
```

## Thông tin thêm:
1. Việc sử dụng ``` @st.cache_resource ```cho việc load models trong Streamlit để tránh reload models mỗi khi người dùng tương tác với giao diện, đồng thời để chia sẻ models giữa các session khác nhau (người dùng khác truy cập ứng dụng, họ sẽ dùng cùng 1 instance của model, không phải tải lại instance mới)

2. Việc sử dụng ```st.rerun()``` sau khi load models xong để cập nhật UI sau khi thay đổi session state

3. Trong ```SemanticChunker```, ```breakpoint_threshold_amount=95``` có nghĩa là chỉ cho phép tạo breakpoint khi similarity < 95%

4. Sử dụng ``` torch_dtype=torch.bfloat16 ``` thay vì ```float32``` để tăng tốc độ inference, giảm 50% lượng memory được sử dụng, giúp tương thích tốt hơn với GPU

5. Trong RAG chain, ``` RunnablePassthrough() ``` là một thành phần trong LangChain Expression Language (LCEL), hoạt động như một bộ chuyển tiếp đơn giản, nhận input đầu vào và truyền nó trực tiếp ra output mà không thực hiện bất kỳ thay đổi hay xử lý nào.

6. Trong ```SemanticChunker```, ```buffer_size=1``` là số lượng câu được so sánh khi tìm breakpoint trong việc chia nhỏ văn bản. Nếu ```buffer_size > 1```, nó sẽ xem xét sự khác biệt ngữ nghĩa tích lũy  hoặc trung bình của một câu với N câu tiếp theo để xác định breakpoint.

7. Tham số ```max_new_tokens=512``` để giới hạn số tokens tối đa model có thể sinh ra.

8. Việc dùng temporary file khi xử lý PDF upload vì ```PyPDFLoader``` yêu cầu file path thật.

9. Prompt ```"rlm/rag-prompt"``` từ LangChain Hub có cấu trúc dạng: System prompt + Context + Question. Tuy nhiên, có thể thay bằng ```PromptTemplate``` để dễ dàng tùy biến và kiểm soát model LLM

10. Việc xử lý ```output.split("Answer:")[1].strip()``` để tách phần trả lời từ format prompt và làm sạch các whitespace thừa.