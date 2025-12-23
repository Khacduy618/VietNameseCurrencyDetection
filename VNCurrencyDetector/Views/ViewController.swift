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

    // --- 2. LOGIC XỬ LÝ ẢNH (FAST/SLOW PATH) ---
    var frameBuffer: [(id: String, conf: Float)] = []
    let maxBufferSize = 20
    
    // Fast Path: Đọc nhanh
    let fastStreakRequired = 4
    let fastConfidence: Float = 0.96
    
    // Slow Path: Bầu cử
    let slowBatchSize = 15
    let slowVoteRequired = 10
    let slowConfidence: Float = 0.85
    
    // Trạng thái
    var lastReadText: String = ""
    var lastReadTime: Date = Date.distantPast
    
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
                    self.frameBuffer.removeAll()
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
            
            // Lọc rác
            if id == "00_background" || conf < 0.7 {
                self.frameBuffer.removeAll()
                self.lastReadText = ""
                self.resultLabel.text = "..."
                self.resultLabel.textColor = .lightGray
                self.debugLabel.text = "\(self.currentModelName) | Nền: \(Int(conf*100))%"
                return
            }

            self.frameBuffer.append((id: id, conf: conf))
            if self.frameBuffer.count > self.maxBufferSize { self.frameBuffer.removeFirst() }
            
            self.debugLabel.text = "\(self.currentModelName) | Buf:\(self.frameBuffer.count) | \(id) \(Int(conf*100))%"

            // Fast Path
            let recentFrames = self.frameBuffer.suffix(self.fastStreakRequired)
            if recentFrames.count >= self.fastStreakRequired {
                let allSameID = recentFrames.allSatisfy { $0.id == id }
                let allHighConf = recentFrames.allSatisfy { $0.conf >= self.fastConfidence }
                
                if allSameID && allHighConf && id != "00_background" {
                    print("🚀 Fast Path: \(id)")
                    self.processFinalResult(id: id, conf: conf)
                    self.frameBuffer.removeAll()
                    return
                }
            }
            
            // Slow Path
            if self.frameBuffer.count >= self.slowBatchSize {
                let counts = self.frameBuffer.reduce(into: [:]) { counts, item in
                    counts[item.id, default: 0] += 1
                }
                
                if let (winnerID, voteCount) = counts.max(by: { $0.value < $1.value }) {
                    let winnerFrames = self.frameBuffer.filter { $0.id == winnerID }
                    let avgConf = winnerFrames.reduce(0) { $0 + $1.conf } / Float(winnerFrames.count)
                    
                    if winnerID != "00_background" && voteCount >= self.slowVoteRequired && avgConf > self.slowConfidence {
                        print("🐢 Slow Path: \(winnerID)")
                        self.processFinalResult(id: winnerID, conf: avgConf)
                    }
                }
                self.frameBuffer.removeFirst(5)
            }
        }
    }
    
    // --- 6. XỬ LÝ KẾT QUẢ & ĐỌC ---
    func processFinalResult(id: String, conf: Float) {
        if id == "00_background" { return }
        
        guard let textToSpeak = moneyMapping[id] else { return }
        
        let percent = Int(conf * 100)
        resultLabel.text = "\(textToSpeak)\n(\(percent)%)"
        resultLabel.textColor = .green
        
        let now = Date()
        if textToSpeak != lastReadText || now.timeIntervalSince(lastReadTime) > 3.0 {
            speak(textToSpeak)
            let generator = UINotificationFeedbackGenerator()
            generator.notificationOccurred(.success)
            lastReadText = textToSpeak
            lastReadTime = now
        }
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