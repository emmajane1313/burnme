import { useState } from "react";
import { StreamPage } from "./pages/StreamPage";
import { Toaster } from "./components/ui/sonner";
import { VideoBackground } from "./components/VideoBackground";
import { VideoBackgroundControls } from "./components/VideoBackgroundControls";
import "./index.css";

const VIDEO_OPTIONS = [
  { id: 1, name: "Puppy Love", src: "/assets/videos/puppylove.mp4" },
  { id: 2, name: "Heart Doves", src: "/assets/videos/heartdoves.mp4" },
  { id: 3, name: "Epic Battle", src: "/assets/videos/epicbattle.mp4" },
];

function App() {
  const [currentVideo, setCurrentVideo] = useState(1);

  const selectedVideo = VIDEO_OPTIONS.find((v) => v.id === currentVideo) || VIDEO_OPTIONS[0];

  return (
    <>
      <VideoBackground videoSrc={selectedVideo.src} />

      <StreamPage
        videoControls={
          <VideoBackgroundControls
            videos={VIDEO_OPTIONS}
            currentVideo={currentVideo}
            onSelectVideo={setCurrentVideo}
          />
        }
      />
      <Toaster />
    </>
  );
}

export default App;
