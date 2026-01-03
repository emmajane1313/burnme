interface VideoOption {
  id: number;
  name: string;
  src: string;
  thumbnail?: string;
}

interface VideoBackgroundControlsProps {
  videos: VideoOption[];
  currentVideo: number;
  onSelectVideo: (id: number) => void;
}

export function VideoBackgroundControls({
  videos,
  currentVideo,
  onSelectVideo,
}: VideoBackgroundControlsProps) {
  return (
    <div className="flex gap-1.5">
      {videos.map((video) => (
        <button
          key={video.id}
          onClick={() => onSelectVideo(video.id)}
          className={`relative w-20 h-12 cursor-pointer overflow-hidden transition-all ${
            currentVideo === video.id
              ? "border-2 border-cyan-500"
              : "border border-transparent hover:border-pink-500/50"
          }`}
        >
          <video
            src={video.src}
            className="w-full h-full object-cover"
            muted
            loop
            autoPlay
            playsInline
          />
          {currentVideo === video.id && (
            <div className="absolute inset-0 bg-cyan-500/20 pointer-events-none" />
          )}
        </button>
      ))}
    </div>
  );
}