import UIKit
import AVFoundation
import Vision
import CoreML



class ViewController: UIViewController, AVCaptureVideoDataOutputSampleBufferDelegate, AVSpeechSynthesizerDelegate {
    
    // 1. Khung ngắm
    let centerFocusView: UIView = {
        let v = UIView()
        v.backgroundColor = .clear
        v.layer.borderColor = UIColor.yellow.cgColor
        v.layer.borderWidth = 2.0
        v.layer.cornerRadius = 10
        // Shadow
        v.layer.shadowColor = UIColor.black.cgColor
        v.layer.shadowOpacity = 0.5
        v.layer.shadowOffset = CGSize(width: 0, height: 2)
        v.layer.shadowRadius = 4
        return v
    }()
    
    // 2. Container chứa kết quả (Nền đen mờ)
    let resultContainerView: UIView = {
        let v = UIView()
        v.backgroundColor = UIColor.black.withAlphaComponent(0.7)
        v.layer.cornerRadius = 15
        v.clipsToBounds = true
        return v
    }()
    
    // 3. Thanh Loading (Stability)
    let stabilityProgressView: UIProgressView = {
        let pv = UIProgressView(progressViewStyle: .bar)
        pv.progress = 0.0
        pv.trackTintColor = .darkGray
        pv.progressTintColor = .green
        pv.layer.cornerRadius = 2
        pv.clipsToBounds = true
        return pv
    }()
    
    // 4. Label Live Info
    let liveInfoLabel: UILabel = {
        let lbl = UILabel()
        lbl.font = UIFont.monospacedDigitSystemFont(ofSize: 14, weight: .medium)
        lbl.textColor = UIColor.lightGray
        lbl.text = "Waiting..."
        lbl.textAlignment = .center
        // Thêm background nhỏ cho dễ đọc
        lbl.backgroundColor = UIColor.black.withAlphaComponent(0.3)
        lbl.layer.cornerRadius = 4
        lbl.clipsToBounds = true
        return lbl
    }()

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
            // ====================================================
            // 1. XỬ LÝ CHỈ SỐ LIVE (TỨC THỜI)
            // ====================================================
            // Ở đây KHÔNG chia trung bình, vì ta muốn biết frame này tốt hay xấu ngay lập tức
            let livePercent = Int(min(conf, 1.0) * 100)
            let rawName = self.moneyMapping[id] ?? id
            
            // Hiển thị Live
            if id == "00_background" {
                 self.liveInfoLabel.text = "Live: Nền/Rác (\(livePercent)%)"
                 self.liveInfoLabel.textColor = .gray
                 self.centerFocusView.layer.borderColor = UIColor.white.withAlphaComponent(0.5).cgColor
            } else {
                 self.liveInfoLabel.text = "Live: \(rawName) (\(livePercent)%)"
                 let isReliable = conf > self.CONFIDENCE_THRESHOLD
                 self.liveInfoLabel.textColor = isReliable ? .green : .orange
                 self.centerFocusView.layer.borderColor = isReliable ? UIColor.green.cgColor : UIColor.yellow.cgColor
            }

            // ====================================================
            // 2. TÍNH TOÁN TRUNG BÌNH CHO KẾT QUẢ CHỐT (STABLE)
            // ====================================================
            
            // Lọc rác
            if id == "00_background" || conf < self.CONFIDENCE_THRESHOLD {
                self.candidateID = nil
                self.candidateCount = 0
                self.candidateConfTotal = 0.0 // Reset tổng
                self.stabilityProgressView.progress = 0.0
                
                self.backgroundCount += 1
                if self.backgroundCount > self.RESET_BG_FRAMES {
                    self.currentStableID = nil
                    self.resultLabel.text = "Đưa tiền vào khung ngắm"
                    self.resultLabel.textColor = .white
                    self.backgroundCount = 0
                }
                return
            }
            
            self.backgroundCount = 0

            // Kiểm tra liên tiếp
            if id == self.candidateID {
                self.candidateCount += 1
                self.candidateConfTotal += conf // Cộng dồn: 0.8 + 0.9 + ...
            } else {
                self.candidateID = id
                self.candidateCount = 1
                self.candidateConfTotal = conf // Bắt đầu đếm lại
            }
            
            // Cập nhật thanh loading
            let progress = Float(self.candidateCount) / Float(self.STABILITY_FRAMES_REQUIRED)
            self.stabilityProgressView.setProgress(progress, animated: true)

            // KHI ĐỦ 15 FRAME -> TÍNH TRUNG BÌNH CỘNG
            if self.candidateCount >= self.STABILITY_FRAMES_REQUIRED {
                
                // [ĐÂY LÀ CHỖ BẠN CẦN]: TỔNG / SỐ LƯỢNG
                let averageConf = self.candidateConfTotal / Float(self.candidateCount)
                
                if id != self.currentStableID {
                    self.currentStableID = id
                    // Gửi giá trị trung bình đi hiển thị
                    self.processFinalResult(id: id, conf: averageConf)
                }
                
                // Giữ trạng thái khóa nhưng vẫn cập nhật trung bình mới nhất nếu muốn
                self.candidateCount = self.STABILITY_FRAMES_REQUIRED
                // Mẹo toán học: Reset tổng về mức trung bình để giữ ổn định cho frame tiếp theo
                self.candidateConfTotal = averageConf * Float(self.STABILITY_FRAMES_REQUIRED)
                
                self.stabilityProgressView.progressTintColor = .systemCyan
            } else {
                self.stabilityProgressView.progressTintColor = .green
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
        view.addSubview(centerFocusView)
        view.addSubview(resultContainerView)
        view.addSubview(liveInfoLabel)
        
        resultContainerView.addSubview(resultLabel)
        resultContainerView.addSubview(stabilityProgressView)
        view.addSubview(debugLabel)
        
        modelSelector.translatesAutoresizingMaskIntoConstraints = false
        centerFocusView.translatesAutoresizingMaskIntoConstraints = false
        resultContainerView.translatesAutoresizingMaskIntoConstraints = false
        liveInfoLabel.translatesAutoresizingMaskIntoConstraints = false
        resultLabel.translatesAutoresizingMaskIntoConstraints = false
        stabilityProgressView.translatesAutoresizingMaskIntoConstraints = false
        debugLabel.translatesAutoresizingMaskIntoConstraints = false
        
        NSLayoutConstraint.activate([
            // 1. Model Selector
            modelSelector.topAnchor.constraint(equalTo: view.safeAreaLayoutGuide.topAnchor, constant: 10),
            modelSelector.centerXAnchor.constraint(equalTo: view.centerXAnchor),
            modelSelector.widthAnchor.constraint(equalToConstant: 320),
            
            // 2. Debug Label
            debugLabel.topAnchor.constraint(equalTo: modelSelector.bottomAnchor, constant: 8),
            debugLabel.centerXAnchor.constraint(equalTo: view.centerXAnchor),
            
            // 3. Center Focus View (SỬA LẠI KÍCH THƯỚC TẠI ĐÂY)
            centerFocusView.centerXAnchor.constraint(equalTo: view.centerXAnchor),
            centerFocusView.centerYAnchor.constraint(equalTo: view.centerYAnchor, constant: -50), // Dịch lên một chút để tránh bị tay che
            // Tăng chiều rộng lên 340 và chiều cao là 220 (hình chữ nhật ngang)
            centerFocusView.widthAnchor.constraint(equalToConstant: 340),
            centerFocusView.heightAnchor.constraint(equalToConstant: 450),
            
            // 4. Live Info Label
            liveInfoLabel.bottomAnchor.constraint(equalTo: centerFocusView.topAnchor, constant: -10),
            liveInfoLabel.centerXAnchor.constraint(equalTo: view.centerXAnchor),
            liveInfoLabel.heightAnchor.constraint(equalToConstant: 24),
            
            // 5. Result Container
            resultContainerView.bottomAnchor.constraint(equalTo: view.safeAreaLayoutGuide.bottomAnchor, constant: -30),
            resultContainerView.leadingAnchor.constraint(equalTo: view.leadingAnchor, constant: 20),
            resultContainerView.trailingAnchor.constraint(equalTo: view.trailingAnchor, constant: -20),
            resultContainerView.heightAnchor.constraint(equalToConstant: 140),
            
            // 6. Bên trong Container
            stabilityProgressView.topAnchor.constraint(equalTo: resultContainerView.topAnchor),
            stabilityProgressView.leadingAnchor.constraint(equalTo: resultContainerView.leadingAnchor),
            stabilityProgressView.trailingAnchor.constraint(equalTo: resultContainerView.trailingAnchor),
            stabilityProgressView.heightAnchor.constraint(equalToConstant: 4),
            
            resultLabel.topAnchor.constraint(equalTo: stabilityProgressView.bottomAnchor, constant: 10),
            resultLabel.leadingAnchor.constraint(equalTo: resultContainerView.leadingAnchor, constant: 10),
            resultLabel.trailingAnchor.constraint(equalTo: resultContainerView.trailingAnchor, constant: -10),
            resultLabel.bottomAnchor.constraint(equalTo: resultContainerView.bottomAnchor, constant: -10)
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