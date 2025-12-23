import UIKit
import AVFoundation
import Vision
import CoreML

class ViewController: UIViewController, AVCaptureVideoDataOutputSampleBufferDelegate, AVSpeechSynthesizerDelegate {

    // --- 1. CẤU HÌNH & BIẾN ---
    var captureSession: AVCaptureSession!
    var visionModel: VNCoreMLModel?
    var currentModelName: String = ""
    
    // Audio
    let synthesizer = AVSpeechSynthesizer()
    
    // UI Components
    var previewLayer: AVCaptureVideoPreviewLayer!
    
    // [CẬP NHẬT] Thanh chọn 3 Model
    let modelSelector: UISegmentedControl = {
        let items = ["ResNet18", "MobileNetV2", "EfficientNet"]
        let sc = UISegmentedControl(items: items)
        sc.selectedSegmentIndex = 0 // Mặc định chọn cái đầu tiên
        sc.backgroundColor = UIColor.white.withAlphaComponent(0.9)
        sc.selectedSegmentTintColor = .systemGreen
        return sc
    }()
    
    // Hiển thị kết quả
    let resultLabel: UILabel = {
        let lbl = UILabel()
        lbl.backgroundColor = UIColor.black.withAlphaComponent(0.7)
        lbl.textColor = .green
        lbl.textAlignment = .center
        lbl.numberOfLines = 0
        lbl.font = .boldSystemFont(ofSize: 32)
        lbl.text = "Sẵn sàng..."
        lbl.layer.cornerRadius = 12
        lbl.layer.masksToBounds = true
        return lbl
    }()
    
    // Debug Info
    let debugLabel: UILabel = {
        let lbl = UILabel()
        lbl.textColor = .yellow
        lbl.font = .monospacedSystemFont(ofSize: 12, weight: .regular)
        lbl.textAlignment = .center
        lbl.text = "Debug Info"
        lbl.backgroundColor = UIColor.black.withAlphaComponent(0.4)
        return lbl
    }()

    // Biến lưu trạng thái hiện tại
    var candidateID: String? = nil        // Ứng cử viên đang xét
    var candidateCount: Int = 0           // Số lần xuất hiện liên tiếp
    var candidateConfTotal: Float = 0.0
    var currentStableID: String? = nil    // Kết quả đã chốt (đang hiển thị)
    
    // Cấu hình độ khó (Tinh chỉnh ở đây)
    // Tăng từ 6 lên 15 (khoảng 0.5 giây) để cực kỳ chắc chắn
    let STABILITY_FRAMES_REQUIRED: Int = 15 
    let CONFIDENCE_THRESHOLD: Float = 0.85 
    
    // Reset nếu gặp background quá lâu
    var backgroundCount: Int = 0
    let RESET_BG_FRAMES: Int = 20          // Đếm số lần gặp nền để reset
    
    // Mapping nhãn
    let moneyMapping: [String: String] = [
        "00_background": "Nền",
        "1k": "Một nghìn",
        "2k": "Hai nghìn",
        "5k": "Năm nghìn",
        "10k": "Mười nghìn",
        "20k": "Hai mươi nghìn",
        "50k": "Năm mươi nghìn",
        "100k": "Một trăm nghìn",
        "200k": "Hai trăm nghìn",
        "500k": "Năm trăm nghìn"
    ]

    // --- LIFECYCLE ---
    override func viewDidLoad() {
        super.viewDidLoad()
        view.backgroundColor = .black
        setupAudio()
        setupUI()
        setupCamera()
        
        synthesizer.delegate = self
        
        // Mặc định load ResNet18
        loadModel(index: 0)
    }
    
    override func viewDidLayoutSubviews() {
        super.viewDidLayoutSubviews()
        previewLayer.frame = view.bounds
    }
    
    // --- 3. LOAD MODEL (ĐÃ CẬP NHẬT CHO 3 MODELS) ---
    func loadModel(index: Int) {
        // Xác định tên hiển thị
        let names = ["Resnet18", "MobileNetV2", "EfficientNetB0"]
        let targetName = names[index]
        
        if currentModelName == targetName { return }
        
        DispatchQueue.global(qos: .userInitiated).async {
            do {
                let config = MLModelConfiguration()
                var coreModel: MLModel?
                
                // [QUAN TRỌNG] Khởi tạo class tương ứng với lựa chọn
                switch index {
                case 0:
                    coreModel = try Resnet18(configuration: config).model
                case 1:
                    coreModel = try MobileNetV2(configuration: config).model
                case 2:
                    // Đảm bảo tên class này khớp với tên file model bạn kéo vào Xcode
                    coreModel = try EfficientNetB0(configuration: config).model
                default:
                    return
                }
                
                guard let model = coreModel else { return }
                let vModel = try VNCoreMLModel(for: model)
                
                DispatchQueue.main.async {
                    self.visionModel = vModel
                    self.currentModelName = targetName
                    self.modelSelector.selectedSegmentIndex = index
                    self.debugLabel.text = "Model: \(targetName)"

                    self.candidateID = nil
                    self.candidateCount = 0
                    self.currentStableID = nil
                    self.backgroundCount = 0

                    self.resultLabel.text = "Sẵn sàng..."
                    self.resultLabel.textColor = .white
                    self.speak("Đã chuyển sang \(targetName)")
                }
            } catch {
                print("Lỗi load model: \(error)")
                DispatchQueue.main.async {
                    self.debugLabel.text = "Thiếu file Model: \(targetName)!"
                }
            }
        }
    }
    
    @objc func modelChanged(_ sender: UISegmentedControl) {
        loadModel(index: sender.selectedSegmentIndex)
    }

    // --- 4. CAMERA DELEGATE ---
    func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection) {
        if synthesizer.isSpeaking { return }
        
        guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer),
              let model = visionModel else { return }
        
        let request = VNCoreMLRequest(model: model) { [weak self] req, _ in
            guard let self = self,
                  let topResult = (req.results as? [VNClassificationObservation])?.first else { return }
            
            self.handleFrameResult(id: topResult.identifier, conf: topResult.confidence)
        }
        
        request.imageCropAndScaleOption = .centerCrop
        try? VNImageRequestHandler(cvPixelBuffer: pixelBuffer, orientation: .right).perform([request])
    }
    
    // --- 5. LOGIC XỬ LÝ THÔNG MINH ---
    func handleFrameResult(id: String, conf: Float) {
        DispatchQueue.main.async {
            
            // 1. LỌC RÁC: Nếu tin cậy thấp hoặc là Background
            if id == "00_background" || conf < self.CONFIDENCE_THRESHOLD {
                // Reset ứng cử viên hiện tại
                self.candidateID = nil
                self.candidateCount = 0
                self.candidateConfTotal = 0.0
                
                // Đếm background để reset màn hình
                self.backgroundCount += 1
                if self.backgroundCount > self.RESET_BG_FRAMES {
                    self.currentStableID = nil
                    self.resultLabel.text = "Sẵn sàng..."
                    self.resultLabel.textColor = .white
                    self.debugLabel.text = "Đang tìm..."
                    self.backgroundCount = 0
                }
                return
            }
            
            // Tìm thấy tiền -> Reset đếm background
            self.backgroundCount = 0

            // 2. KIỂM TRA TÍNH LIÊN TIẾP
            if id == self.candidateID {
                // Nếu GIỐNG frame trước -> Cộng dồn
                self.candidateCount += 1
                self.candidateConfTotal += conf // Cộng điểm để tính trung bình
            } else {
                // Nếu KHÁC frame trước -> Reset, bắt đầu đếm lại từ số 1
                self.candidateID = id
                self.candidateCount = 1
                self.candidateConfTotal = conf
            }
            
            // Debug cho dev xem
            self.debugLabel.text = "Scan: \(self.moneyMapping[id] ?? id) | Giữ yên: \(self.candidateCount)/\(self.STABILITY_FRAMES_REQUIRED)"

            // 3. CHỐT KẾT QUẢ (Khi đủ 15 frames liên tiếp)
            if self.candidateCount >= self.STABILITY_FRAMES_REQUIRED {
                
                // Tính trung bình cộng độ tin cậy (FIX LỖI 800%)
                let averageConf = self.candidateConfTotal / Float(self.candidateCount)
                
                // Chỉ cập nhật nếu kết quả KHÁC với cái đang hiện trên màn hình
                if id != self.currentStableID {
                    self.currentStableID = id
                    self.processFinalResult(id: id, conf: averageConf)
                } else {
                    // Nếu vẫn là tờ tiền cũ, chỉ cập nhật lại % cho chuẩn (nếu muốn)
                    // self.updateConfidenceDisplay(conf: averageConf) 
                }
                
                // Giữ bộ đếm ở mức max để tránh tràn số, nhưng vẫn giữ ID này là candidate
                self.candidateCount = self.STABILITY_FRAMES_REQUIRED
                // Reset total để tránh cộng dồn vô tận, giữ lại giá trị trung bình hiện tại
                self.candidateConfTotal = averageConf * Float(self.STABILITY_FRAMES_REQUIRED)
            }
        }
    }


    // --- 6. XỬ LÝ KẾT QUẢ & ĐỌC ---
    func processFinalResult(id: String, conf: Float) {
        guard let textToSpeak = moneyMapping[id] else { return }
        
        
        resultLabel.text = "\(textToSpeak)"
        
        // Màu sắc
        if ["1k", "2k", "5k"].contains(id) {
            resultLabel.textColor = .yellow
        } else {
            resultLabel.textColor = .green
        }
        
        // Rung & Đọc
        let generator = UINotificationFeedbackGenerator()
        generator.notificationOccurred(.success)
        speak(textToSpeak)
    }
    
    // --- 7. AUDIO SETUP ---
    func setupAudio() {
        try? AVAudioSession.sharedInstance().setCategory(.playback, mode: .default)
        try? AVAudioSession.sharedInstance().setActive(true)
    }
    
    func speak(_ text: String) {
        let utterance = AVSpeechUtterance(string: text)
        utterance.voice = AVSpeechSynthesisVoice(language: "vi-VN")
        utterance.rate = 0.5
        synthesizer.speak(utterance)
    }
    
    // --- 8. UI LAYOUT ---
    func setupUI() {
        view.addSubview(modelSelector)
        view.addSubview(resultLabel)
        view.addSubview(debugLabel)
        
        modelSelector.translatesAutoresizingMaskIntoConstraints = false
        resultLabel.translatesAutoresizingMaskIntoConstraints = false
        debugLabel.translatesAutoresizingMaskIntoConstraints = false
        
        NSLayoutConstraint.activate([
            modelSelector.topAnchor.constraint(equalTo: view.safeAreaLayoutGuide.topAnchor, constant: 10),
            modelSelector.centerXAnchor.constraint(equalTo: view.centerXAnchor),
            modelSelector.widthAnchor.constraint(equalToConstant: 320), // Tăng chiều rộng để chứa 3 nút
            
            debugLabel.topAnchor.constraint(equalTo: modelSelector.bottomAnchor, constant: 10),
            debugLabel.centerXAnchor.constraint(equalTo: view.centerXAnchor),
            
            resultLabel.bottomAnchor.constraint(equalTo: view.safeAreaLayoutGuide.bottomAnchor, constant: -50),
            resultLabel.leadingAnchor.constraint(equalTo: view.leadingAnchor, constant: 20),
            resultLabel.trailingAnchor.constraint(equalTo: view.trailingAnchor, constant: -20),
            resultLabel.heightAnchor.constraint(equalToConstant: 120)
        ])
        
        modelSelector.addTarget(self, action: #selector(modelChanged), for: .valueChanged)
    }
    
    func setupCamera() {
        captureSession = AVCaptureSession()
        captureSession.sessionPreset = .hd1280x720
        
        guard let device = AVCaptureDevice.default(for: .video),
              let input = try? AVCaptureDeviceInput(device: device) else { return }
        
        if captureSession.canAddInput(input) {
            captureSession.addInput(input)
        }
        
        let output = AVCaptureVideoDataOutput()
        output.setSampleBufferDelegate(self, queue: DispatchQueue(label: "videoQueue"))
        output.alwaysDiscardsLateVideoFrames = true
        
        if captureSession.canAddOutput(output) {
            captureSession.addOutput(output)
        }
        
        previewLayer = AVCaptureVideoPreviewLayer(session: captureSession)
        previewLayer.videoGravity = .resizeAspectFill
        view.layer.insertSublayer(previewLayer, at: 0)
        
        DispatchQueue.global(qos: .userInitiated).async { self.captureSession.startRunning() }
    }
}