# Báo cáo bài thực hành Ngày 1 – Đọc prediction YOLO11 để hiểu định dạng nhãn

**Ngày chạy:** 6 thg 11, 2026 17:12

**Runtime Colab:** GPU T4

**Python / PyTorch / Ultralytics:** Python: 3.13.16 / PyTorch: 2.11.0+cu130 / Ultralytics: 8.4.145

**Checkpoint:** `yolo11n-cls.pt`, `yolo11n.pt`, `yolo11n-seg.pt`

**Ngưỡng detection/segmentation dùng khi tạo JSON:** `0.35` (chỉ đổi nếu GV cho phép; nếu đổi hãy ghi rõ)

**Thay đổi so với notebook nguồn:** Không / mô tả rõ thay đổi

> **Cách dùng mẫu này:** copy/điền trên máy, lưu thành `REPORT.md`, rồi **upload** vào panel Files
> của Colab (notebook không tự tạo file này). Ô cuối Colab sẽ đóng ZIP cùng `day1_lab_outputs/`.
> Giải nén và đưa vào `report/` trên GitHub, rồi nộp **link repo** trên LMS.
> Không ghi họ tên, MSSV, email, số điện thoại hoặc dữ liệu cá nhân khác vào báo cáo/output;
> họ tên/MSSV chỉ nằm trong tên repository.

### Từ cần nhớ (đọc nhanh trước khi viết)

| Từ | Nghĩa ngắn |
| --- | --- |
| Prediction | Kết quả model vừa đoán ra; chưa phải nhãn đúng |
| Ground truth | Nhãn do người gán theo quy tắc |
| Taxonomy | Danh sách tên lớp có sẵn kèm model (vd. ImageNet-1K, COCO-80) |
| Checkpoint | File model `.pt` đã huấn luyện sẵn |
| Guideline | Quy tắc gán nhãn do người viết |
| Annotator | Người gán nhãn |
| Reviewer / QC | Người kiểm tra nhãn |
| Escalation | Ca khó → hỏi Lab Coach, không tự bịa |
| Model score | Số “tin” của model; không phải điểm chất lượng nhãn người |

## 1. Phân loại ảnh

**Mở file:** `day1_lab_outputs/classification_predictions.json` và `visuals/classification_top5.png`.  
Chỉ dùng ảnh `sample_id = traffic`, lấy dòng `rank = 1`.

1. Chép record hạng 1: `class_id`=468 ; `class_name`=`cab` ; `rank`=1 ; `score`=0.510915 ; `taxonomy_name`=ImageNet-1K
2. Record này nói về **cả bức ảnh** hay về **một vật thể riêng** trong ảnh? Vì sao bạn nghĩ vậy?
```
Record này nói về cả bức ảnh, vì đây là output của classification, trong record không có thông tin về vị trí của vật thể.
```
3. Danh sách tên lớp (class list) do đâu mà có: model tự nghĩ ra, hay đi kèm taxonomy/checkpoint?
```
Danh sách tên lớp này đi kèm theo taxonomy và checkpoint của mô hình
```
4. Vì sao cần ghi cả `class_id`, `class_name` và `taxonomy_name`, không chỉ ghi mỗi tên lớp?
```
class_id để có thể cho máy tính xác định dễ dàng
class_name để con người có thể dễ nhận diện ngay thay vì phải tra cứu class_id
taxonomy_name để xác định được lớp thuộc bộ dữ liệu
```
5. Nếu trong ảnh có nhiều vật / nhiều chủ đề, guideline cần nói rõ điều gì để người gán nhãn chọn **một** lớp?
```
- Chọn vật thể chiếm diện tích lớn nhất, nằm ở vị trí trung tâm
- Ngữ cảnh bao quát, Ưu tiên nhãn bối cảnh tổng thể
- Quy định thứ tự ưu tiên cố định
- Chọn nhãn miêu tả chi tiết, rõ ràng nhất thay vì nhãn chung chung
- Nếu các vật thể nổi bật ngang nhau, ưu tiên vật nằm gần tâm ảnh nhất.
```
6. Vì sao không được lấy `score` của model làm ground truth?
```
- Score chỉ thể hiện mức độ chắc chắn của mô hình
- Mô hình vẫn dự đoán sai: Điểm số cao (ví dụ: 0.79) vẫn có thể sai hoàn toàn nếu ảnh đánh lừa mô hình hoặc nằm ngoài phân phối huấn luyện
- Mô hình có thể tự tin thái quá vào các nhãn quen thuộc
- Dữ liệu chuẩn gốc phải đến từ con người gán nhãn thực tế
```
## 2. Phát hiện vật thể

**Mở file:** `day1_lab_outputs/detection_predictions.json` và `visuals/detection_predictions.png`.  
Chỉ dùng ảnh `sample_id = kitchen`. Chọn **một** record.  
Số lượng prediction theo ngưỡng: xem output ô so sánh `0.20` / `0.35` / `0.60` trong notebook.

1. Chép một record: `class_name`=`person` ; `score`=0.912625 ; `bbox_xyxy`=[385.33,69.24,498.92,348.92] ; `bbox_width`=113.58 ; `bbox_height`=279.68
2. Giải thích box bằng lời thường: góc trên-trái ở đâu, góc dưới-phải ở đâu, rộng/cao bao nhiêu pixel?
```
- Góc trên-trái (385.33, 69.24)
- Góc dưới phải (498.92, 348.92)
- Rộng 113.58 pixel
- Cao 279.68 pixel
```
3. Chọn hai ngưỡng (vd. 0.20 và 0.60). Mỗi ngưỡng có bao nhiêu prediction?
```
threshold=0.20: 17 vật thể
threshold=0.60: 6 vật thể
```
4. Khi hạ ngưỡng (giữ nhiều hộp hơn), người **reviewer** phải làm thêm việc gì?
```
Khi hạ ngưỡng, số lượng hộp tăng lên đồng nghĩa với nhiều hộp độ tin cậy thấp xuất hiện, lúc này người reviewer cần
- Lọc và loại bỏ các kết quả dương tính giả
- Xử lý khối lượng công việc lớn hơn
- Xác minh các vật thể khó nhận diện
```
5. Viết **một quy tắc box** để người gán nhãn làm theo (áp dụng được cho nhiều ảnh, không chỉ ảnh này).
```
- Vẽ ôm khít đối tượng (Tight Fit):
Box phải bao trọn toàn bộ ranh giới ngoài cùng của vật thể (tính cả các phần nhô ra).
Không để quá nhiều khoảng trống nền (background) thừa bên trong box và không để box cắt xén lẹm vào vật thể.

- Xử lý vật bị che khuất (Occlusion):
Chỉ vẽ box bao lấy phần diện tích thực tế đang nhìn thấy của đối tượng bị che một phần.
Bỏ qua đối tượng nếu bị che khuất quá nhiều (>70-80%) hoặc quá mờ không thể nhận diện.

- Gán đúng nhãn (Labeling):
Chọn chính xác tên lớp (class name) theo đúng danh mục quy định của dự án (ví dụ: phân biệt rõ car và bus, bowl và cup).

- Lọc vật thể quá nhỏ / ở xa:
Chỉ gán nhãn cho những vật thể mà mắt thường có thể nhận dạng rõ ràng hình thù.
Bỏ qua các vật thể quá nhỏ hoặc quá mờ ở xa để tránh gây nhiễu dữ liệu.
```
6. Trên ảnh `kitchen`, chỉ ra **một** vật bị che khuất hoặc cắt mép. Bạn sẽ viết thành quy tắc trong guideline, hay đưa lên hỏi Lab Coach (escalation)? Vì sao?
```
Quyết định: Đưa lên hỏi Lab Coach (escalation) thay vì tự ý bổ sung ngay vào guideline.

Tính chất phân ranh giới (Edge cases): Các trường hợp vật thể bị cắt mép ngay sát biên ảnh (như tràn viền 0 pixel hoặc bị lẹm một phần nhỏ ở rìa) thường có quy tắc xử lý chung, nhưng với các mô hình phát hiện đối tượng (như YOLO), việc có nên giữ lại hay cắt gọt bounding box bám sát biên hay không phụ thuộc rất lớn vào chính sách dữ liệu (data policy) của từng dự án cụ thể.

Tránh mâu thuẫn tiêu chuẩn: Nếu tự ý viết quy tắc riêng, người gán nhãn có thể gặp tình huống mâu thuẫn giữa việc "có nên kéo box chạm mép ảnh hay thụt lùi vào trong". Việc escalate lên Lab Coach giúp làm rõ chuẩn thống nhất cho toàn bộ team (ví dụ: Box có bắt buộc phải kịch kim biên ảnh hay cắt tại điểm thực tế nhìn thấy).

Đảm bảo tính nhất quán của tập dữ liệu: Lab Coach sẽ đưa ra phán quyết chuẩn xác nhất để cập nhật vào tài liệu guideline chung, giúp tất cả các reviewer và labeler áp dụng đồng bộ, tránh tình trạng mỗi người tự hiểu một kiểu làm nhiễu nhãn mô hình.
```

## 3. Phân đoạn theo từng đối tượng

**Mở file:** `day1_lab_outputs/segmentation_predictions.json` và `visuals/segmentation_prediction.png`.  
Chỉ dùng ảnh `sample_id = kitchen`. Chọn **một** record.

1. Chép một record: `instance_id`=kitchen-001 ; `class_name`=person ; `score`=0.899318 ; số điểm polygon=`polygon_point_count`=348 ; 3–5 điểm đầu của `polygon_xy`=[446.0,70.0],[445.0,71.0],[444.0,71.0],[443.0,72.0],[442.0,72.0]
2. So với box, polygon cho thêm thông tin gì về hình dạng vật thể?
```
Polygon mô tả đường viền của vật thể sát hơn, kể cả các phần lồi lõm và hình dạng không đều. Box chỉ cho biết khung chữ nhật bao quanh vật thể, nên không thể hiện chính xác phần nào bên trong khung thuộc về vật thể. 
```
3. `instance_id` dùng để làm gì? Nó **không** phải class ID và **không** phải tracking ID — bạn hiểu câu này thế nào?
```
instance_id dùng để phân biệt rạch ròi các đối tượng thuộc cùng một loại (class) xuất hiện trong cùng một khung hình
```
4. Viết **một quy tắc vẽ biên mask** để người gán nhãn làm theo.
```
Quy tắc vẽ mask theo sát đường viền thực tế của phần vật thể nhìn thấy, đặt các điểm biên ngay tại ranh giới với nền và tuyệt đối không lấy thêm nền hay tự ý đoán vẽ phần bị che khuất. Đối với vùng biên mờ, hãy đặt các điểm biên tại đường chuyển tiếp trung bình giữa vật thể và nền để đảm bảo độ chính xác, không lẹm sâu vào trong cũng như không lan rộng ra ngoài.
```
5. Trên ảnh `kitchen`, chỉ ra **một** chỗ biên mờ / hai vật chạm nhau / bị che. Bạn dùng guideline hay escalation? Vì sao?
```
Sử dụng Guideline trước do Guideline đã hướng dẫn rõ 
```
## 4. Vòng đời và kiểm tra chất lượng

Lab hôm nay chỉ **xem prediction**. Chuỗi dưới đây là vòng đời production để bạn nối các khái niệm:

`ảnh thô → guideline → ground truth → huấn luyện → prediction → QC/rework`

Điền bảng (viết ngắn, có dẫn chứng từ JSON/PNG khi nói về lỗi):

| Tác vụ | Ground truth trông như thế nào? (đơn vị nhãn) | Lỗi hoặc chỗ khó quyết định bạn thấy trên ảnh/JSON | Annotator làm gì? | Reviewer xem gì? |
| --- | --- | --- | --- | --- |
| Phân loại ảnh | Một nhãn cho toàn cảnh: traffic  | mô hình xếp cab cao nhất với score 0.510915, tiếp theo là minibus 0.164284 | Xem ảnh và guideline để chọn nhãn ground truth phù hợp; không sao chép prediction của mô hình nếu chưa xác minh | Kiểm tra nhãn có phù hợp với ảnh và taxonomy không, và cách áp dụng guideline có nhất quán không. |
| Phát hiện vật thể | Mỗi vật thể đều đc gán nhãn và có bounding box | Trong ảnh kitchen, nhiều chiếc bát ở gần nhau nên box có thể chồng lấn hoặc khó xác định ranh giới. JSON cũng có box oven sát mép trái ảnh (x1 = 0.11). Đây là điểm cần kiểm tra, chưa thể kết luận là lỗi chỉ dựa vào prediction. | Đối chiếu từng vật thể nhìn thấy với guideline; vẽ box ôm sát vật thể và xử lý vật bị che/cắt mép theo quy tắc đã thống nhất. | Kiểm tra có bỏ sót hoặc gán trùng vật thể không, nhãn lớp đúng không, và box có bao sát đúng phần vật thể không. |
| Instance segmentation | Mỗi vật đều đc gán nhãn và một mask/polygon riêng bám theo phần vật thể nhìn thấy | dining table phủ gần như toàn bộ mặt bàn và một phần chân/bên dưới, nơi ranh giới với vật dụng xung quanh hoặc nền khó thấy. Prediction cho instance này polygon_point_count = 598. Cần đối chiếu ảnh gốc và guideline để quyết định biên chính xác. | Vẽ một mask riêng cho từng vật thể, theo sát phần nhìn thấy được; không gộp các vật cùng lớp hoặc lấy nền vào mask. Nếu biên không rõ và guideline chưa quy định, hỏi Lab Coach. | Kiểm tra đúng lớp và từng instance, mask có bám ranh giới vật thể không, có bỏ sót phần nhìn thấy hoặc lấy nhầm nền/vật khác không. |

## 5. An toàn dữ liệu

1. Quy tắc bảo vệ dữ liệu của bạn cho bài lab này là gì? (viết 1 câu)
```
Chỉ sử dụng dữ liệu trong phạm vi bài lab, không chia sẻ hoặc tải lên nơi khác và không đưa thông tin cá nhân vào báo cáo.
```
2. Nếu gặp ảnh hoặc dữ liệu không đúng phạm vi, bạn sẽ dừng và báo cho ai?
```
Tôi sẽ dừng xử lý, không tải xuống hay chia sẻ thêm, rồi báo cho Lab Coach hoặc người phụ trách bài lab.
```
## 6. Checklist trước khi nộp link repo

Kiểm tra trên GitHub trong thư mục `report/`:

- [ ] `REPORT.md` (file này đã điền)
- [ ] `day1_lab_outputs/classification_predictions.json`
- [ ] `day1_lab_outputs/detection_predictions.json`
- [ ] `day1_lab_outputs/segmentation_predictions.json`
- [ ] `day1_lab_outputs/IMAGE_ATTRIBUTION.md`
- [ ] `day1_lab_outputs/visuals/classification_top5.png`
- [ ] `day1_lab_outputs/visuals/detection_predictions.png`
- [ ] `day1_lab_outputs/visuals/segmentation_prediction.png`
- [ ] Ô validation cuối notebook báo `PASS`
- [ ] Không có họ tên, MSSV hoặc dữ liệu nhạy cảm trong báo cáo/output
- [ ] Đã nộp **link repository** trên LMS
