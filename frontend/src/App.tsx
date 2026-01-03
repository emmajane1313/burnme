import { useState } from "react";
import { StreamPage } from "./pages/StreamPage";
import { Toaster } from "./components/ui/sonner";
import { VideoBackground } from "./components/VideoBackground";
import { VideoBackgroundControls } from "./components/VideoBackgroundControls";
import { StatusBar } from "./components/StatusBar";
import "./index.css";

const VIDEO_OPTIONS = [
  { id: 1, name: "Puppy Love", src: "/assets/videos/puppylove.mp4" },
  { id: 2, name: "Heart Doves", src: "/assets/videos/heartdoves.mp4" },
  { id: 3, name: "Epic Battle", src: "/assets/videos/epicbattle.mp4" },
];

function App() {
  const [currentVideo, setCurrentVideo] = useState(1);
  const [stats, setStats] = useState({ fps: 0, bitrate: 0 });

  const selectedVideo = VIDEO_OPTIONS.find((v) => v.id === currentVideo) || VIDEO_OPTIONS[0];

  return (
    <div className="h-screen flex flex-col">
      <VideoBackground videoSrc={selectedVideo.src} />

      <div className="flex-1 overflow-hidden">
        <StreamPage
          videoControls={
            <VideoBackgroundControls
              videos={VIDEO_OPTIONS}
              currentVideo={currentVideo}
              onSelectVideo={setCurrentVideo}
            />
          }
          onStatsChange={setStats}
        />
      </div>

      <StatusBar
        className="mac-translucent-ruby mt-2"
        fps={stats.fps}
        bitrate={stats.bitrate}
        videoControls={
          <VideoBackgroundControls
            videos={VIDEO_OPTIONS}
            currentVideo={currentVideo}
            onSelectVideo={setCurrentVideo}
          />
        }
      />
      <Toaster />
    </div>
  );
}

export default App;
