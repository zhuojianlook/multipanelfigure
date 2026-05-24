// ─────────────────────────────────────────────────────────────
//  mpfb-recorder — ScreenCaptureKit window recorder helper
// ─────────────────────────────────────────────────────────────
//  Records a SINGLE application window (not the whole screen) using
//  ScreenCaptureKit, encoding to H.264 mp4 via AVAssetWriter.  Unlike
//  the ffmpeg/avfoundation path (which can only grab a whole display
//  and crop a fixed rect), SCK captures the window's own content and
//  follows it if it moves/resizes, and excludes other windows.
//
//  Usage:
//    mpfb-recorder --pid <ownerPID> --out <path.mp4> [--fps 30] [--title <substr>]
//
//  The app passes its OWN process id (--pid); we pick that process's
//  largest on-screen window (optionally matching --title).  Capture
//  starts immediately and runs until SIGINT/SIGTERM, at which point we
//  finalize the mp4 (write the moov atom) and exit 0.  Any failure
//  (no permission, no window, writer error) exits non-zero with an
//  ERROR: line on stderr, which the app surfaces to the user.
//
//  Requires macOS 14+ (SCContentFilter.pointPixelScale / contentRect).
//  On older macOS the app falls back to the ffmpeg path.
import Foundation
import AppKit
import ScreenCaptureKit
import AVFoundation
import CoreMedia
import CoreGraphics

func argValue(_ name: String) -> String? {
    let a = CommandLine.arguments
    if let i = a.firstIndex(of: name), i + 1 < a.count { return a[i + 1] }
    return nil
}

func die(_ msg: String, _ code: Int32) -> Never {
    FileHandle.standardError.write(("ERROR: " + msg + "\n").data(using: .utf8)!)
    exit(code)
}

guard let outPath = argValue("--out") else { die("--out <path> required", 2) }
let targetPid = Int(argValue("--pid") ?? "") ?? -1
let titleHint = argValue("--title")
let fps = Int32(argValue("--fps") ?? "30") ?? 30

final class Recorder: NSObject, SCStreamOutput, SCStreamDelegate {
    let outURL: URL
    let fps: Int32
    var stream: SCStream?
    var writer: AVAssetWriter?
    var input: AVAssetWriterInput?
    var started = false
    var finishing = false
    let queue = DispatchQueue(label: "mpfb.recorder.output")

    init(outURL: URL, fps: Int32) { self.outURL = outURL; self.fps = fps }

    func start(pid: Int, titleHint: String?) {
        SCShareableContent.getExcludingDesktopWindows(false, onScreenWindowsOnly: true) { content, error in
            if let error = error { die("Screen Recording permission / shareable content failed: \(error.localizedDescription)", 3) }
            guard let content = content else { die("no shareable content", 3) }
            let candidates = content.windows.filter { w in
                guard let app = w.owningApplication else { return false }
                if pid > 0 && app.processID != pid_t(pid) { return false }
                return w.isOnScreen && w.frame.width > 40 && w.frame.height > 40
            }
            var chosen: SCWindow?
            if let t = titleHint, !t.isEmpty {
                chosen = candidates.first { ($0.title ?? "").contains(t) }
            }
            if chosen == nil {
                chosen = candidates.max { ($0.frame.width * $0.frame.height) < ($1.frame.width * $1.frame.height) }
            }
            guard let window = chosen else { die("no capturable window found for pid \(pid)", 4) }
            self.beginCapture(window: window)
        }
    }

    func beginCapture(window: SCWindow) {
        let filter = SCContentFilter(desktopIndependentWindow: window)
        let config = SCStreamConfiguration()
        let scale = CGFloat(filter.pointPixelScale)
        // Even pixel dimensions (H.264 / yuv420p requirement).
        var w = Int((filter.contentRect.width * scale).rounded()); w -= w % 2
        var h = Int((filter.contentRect.height * scale).rounded()); h -= h % 2
        config.width = max(w, 2)
        config.height = max(h, 2)
        config.minimumFrameInterval = CMTime(value: 1, timescale: fps)
        config.showsCursor = true
        config.pixelFormat = kCVPixelFormatType_32BGRA
        config.queueDepth = 6

        do {
            try? FileManager.default.removeItem(at: outURL)
            let writer = try AVAssetWriter(outputURL: outURL, fileType: .mp4)
            let settings: [String: Any] = [
                AVVideoCodecKey: AVVideoCodecType.h264,
                AVVideoWidthKey: config.width,
                AVVideoHeightKey: config.height,
            ]
            let input = AVAssetWriterInput(mediaType: .video, outputSettings: settings)
            input.expectsMediaDataInRealTime = true
            if writer.canAdd(input) { writer.add(input) }
            self.writer = writer
            self.input = input
        } catch {
            die("asset writer init failed: \(error.localizedDescription)", 5)
        }

        let stream = SCStream(filter: filter, configuration: config, delegate: self)
        do {
            try stream.addStreamOutput(self, type: .screen, sampleHandlerQueue: queue)
        } catch {
            die("addStreamOutput failed: \(error.localizedDescription)", 6)
        }
        self.stream = stream
        stream.startCapture { error in
            if let error = error { die("startCapture failed: \(error.localizedDescription)", 7) }
            FileHandle.standardError.write("RECORDING\n".data(using: .utf8)!)
        }
    }

    func stream(_ stream: SCStream, didOutputSampleBuffer sampleBuffer: CMSampleBuffer, of type: SCStreamOutputType) {
        guard type == .screen, !finishing, sampleBuffer.isValid else { return }
        guard let arr = CMSampleBufferGetSampleAttachmentsArray(sampleBuffer, createIfNecessary: false) as? [[SCStreamFrameInfo: Any]],
              let attach = arr.first,
              let raw = attach[SCStreamFrameInfo.status] as? Int,
              let status = SCFrameStatus(rawValue: raw), status == .complete else { return }
        guard let writer = writer, let input = input else { return }
        if !started {
            if writer.status == .unknown {
                writer.startWriting()
                writer.startSession(atSourceTime: CMSampleBufferGetPresentationTimeStamp(sampleBuffer))
                started = true
            }
        }
        if started && input.isReadyForMoreMediaData {
            input.append(sampleBuffer)
        }
    }

    func stop() {
        if finishing { return }
        finishing = true
        stream?.stopCapture { _ in
            self.input?.markAsFinished()
            if let writer = self.writer, writer.status == .writing {
                writer.finishWriting { exit(0) }
            } else {
                exit(0)
            }
        }
    }

    func stream(_ stream: SCStream, didStopWithError error: Error) {
        die("stream stopped: \(error.localizedDescription)", 8)
    }
}

let rec = Recorder(outURL: URL(fileURLWithPath: outPath), fps: fps)

// ScreenCaptureKit (and the CoreGraphics it relies on) need a connection to the
// WindowServer. A bare CLI tool doesn't establish one automatically and crashes
// with: Assertion failed: (did_initialize) … CGS_REQUIRE_INIT … CGInitialization.c.
// Bringing up a headless (dock-less, menu-bar-less) NSApplication initializes
// CoreGraphics so SCK can run from this helper.
let nsApp = NSApplication.shared
nsApp.setActivationPolicy(.accessory)

// Graceful stop on SIGINT / SIGTERM: ignore the default action and route
// to a dispatch source so we can finalize the mp4 before exiting.
signal(SIGINT, SIG_IGN)
signal(SIGTERM, SIG_IGN)
let sigint = DispatchSource.makeSignalSource(signal: SIGINT, queue: .main)
sigint.setEventHandler { rec.stop() }
sigint.resume()
let sigterm = DispatchSource.makeSignalSource(signal: SIGTERM, queue: .main)
sigterm.setEventHandler { rec.stop() }
sigterm.resume()

// Start capture once the run loop is up, then run the AppKit main loop (which
// services the SCK completion handlers + the signal dispatch sources).
DispatchQueue.main.async { rec.start(pid: targetPid, titleHint: titleHint) }
nsApp.run()
