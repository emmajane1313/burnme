import { Card, CardContent, CardHeader, CardTitle } from "./ui/card";

export function AboutPanel({ className = "" }: { className?: string }) {
  return (
    <Card className={`h-full flex flex-col mac-translucent-ruby ${className}`}>
      <CardHeader className="flex-shrink-0">
        <CardTitle className="text-base font-medium text-white">
          Why do you want to get burned?
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-4 overflow-y-auto flex-1 p-4 text-sm text-white/90">
        live video synth.
        <br />
        live video samizdat.
        <br />
        <br />
        encrypt data, or encrypt perception.
        <br />
        <br />
        video is usually the opposite of private. it is one of the most exposed
        mediums we have, the easiest to duplicate, the easiest to circulate, the
        hardest to contain. anyone can record you, repost you, archive you,
        watch you without context, without consent, without you ever knowing.
        the question is not just how to keep something yours anymore. the
        question is what kind of machine lets you exist in public without being
        completely legible.
        <br />
        <br />
        maybe you don't need to disappear.
        <br />
        <br />
        but you do need to become harder to read.
        <br />
        <br />
        <br />
        not privacy through absence, but privacy through interpretation.
        <br />
        <br />
        this is not about replacing cryptography. it is about letting
        cryptography live inside a medium that already moves through culture on
        its own. instead of pushing everything down into invisible layers of
        math and code, the surface itself starts to carry the work. what you see
        becomes part of how meaning is shaped. layered, shifting, slightly off,
        suspicious in that CYOA escaping through the mirrors of victorian halls
        at midnight.
        <br />
        <br />
        and now its not just an mp4.
        <br />
        <br />
        it becomes an mp4p.
        <br />
        <br />
        an mp4-privy.
        <br />
        <br />
        burn me while i’m hot does not pull meaning away from view. it lets it
        sit right in front of everyone, but in a form that refuses to settle
        into something readable on its own. the content moves freely. the
        meaning does not. the synth is no longer just decoration. it burns
        itself into the video.
        <br />
        <br />
        so the question shifts.
        <br />
        <br />
        not how do you regain the keys to agentic privacy.
        <br />
        but how do you burn yourself into the surface in a way that stays yours.
        <br />
        <br />
        why live video?
        <br />
        <br />
        live video is computationally much more demandingly random. real-time
        synthesis introduces variation at the level of frame timing, internal
        state, and render order. Those differences are small, but they are
        stable enough to matter cryptographically and unstable enough to resist
        replay.
        <br />
        <br />
        here, the frames of the live synth are treated as input to the
        encryption process. each burn frame becomes part of the key schedule for
        the visual payloads. The exact bytes of the frame are hashed and folded
        into per-frame key material. That means the encryption context is bound
        not only to abstract settings like prompt or seed, but to the concrete
        result of a specific render that happened at a specific moment.
        <br />
        <br />
        live video provides a time-bound, non-replayable visual source that the
        encryption pipeline can depend on, making restoration inseparable from
        the exact circumstances in which the burn was created.
        <br />
        <br />
        how it works
        <br />
        1. upload a video where you are clearly visible in frame
        <br />
        2. sam3 runs on the footage and builds a frame by frame mask of you
        <br />
        3. pick a y2k style and prompt set to visually transform with loud
        colours, patterns, and overlays
        <br />
        4. hit burn to create the synth output and bind the masked pixels
        through per frame visual cipher keying
        <br />
        5. export the mp4p container together with the separate key file used
        for decryption later
        <br />
        6. load the mp4p into the player to view the burned version of the video
        <br />
        7. import the key file to reverse the visual cipher and reconstruct the
        masked areas
        <br />
        <br />
        how encryption and decryption happen
        <br />
        encryption starts with a secret input, keyMaterial, and a fixed set of
        public visual-cipher metadata: prompt, stable params, seed, and mask
        configuration. from these, the system derives a base_key by hashing
        keyMaterial, the prompt, the params, and the seed with sha256.
        <br />
        <br />
        the burn video is rendered and its frames become cryptographic input.
        for each frame, the exact frame bytes are hashed to produce a
        frame_hash. this hash and the frame index are combined with the base key
        using hmac-sha256 to derive a unique frame_key.
        <br />
        <br />
        from each frame_key, a keystream is generated by running hmac-sha256
        with incrementing counters and truncating the output to the full frame
        rgb length (height × width × 3). this keystream is xor-ed with the
        original video’s rgb pixels inside the mask. pixels outside the mask are
        zeroed. the result is encrypted visual noise tied to that specific burn
        frame.
        <br />
        <br />
        each encrypted frame is stored as a png with rgba channels, where rgb
        contains the encrypted pixels and alpha contains the mask. these frames
        are base64-encoded and saved in the mp4p file as encryptedMaskFrames,
        with a maskFrameIndexMap linking each payload to its burn frame. in
        parallel, the original and synthed videos are encrypted with aes-gcm
        using a key derived from keyMaterial via pbkdf2, with salt, iv, and auth
        tag stored in mp4p metadata.
        <br />
        <br />
        decryption requires the mp4p file, the key file, and the exact burn
        frames. the system re-derives the same base_key, recomputes each
        frame_key, regenerates the keystream, and xor-s it with the stored
        payloads to recover the masked pixels. these pixels are composited back
        onto the burn frames using the stored mask and mask mode.
        <br />
        <br />
        if the burn frames and key material match, the masked regions resolve.
        if either differs, the keystream diverges and the output stays noise.
        <br />
        <br />
        some mp4p specifics
        <br />
        the format is a cryptographic container that holds encrypted media
        streams, visual-cipher payloads, and the metadata needed to recompute
        them. the original video exists inside it as aes-gcm encrypted bytes. it
        is not extractable without key material, existing only as per-frame
        visual payloads that are mathematically bound to a specific burn video
        and a key file.
        <br />
        <br />
        without that exact burn context and the matching key material, there is
        nothing meaningful to decode, no partial frames to recover, no
        underlying stream to dump. mp4p stores encrypted video bytes plus the
        conditions under which a video can be reconstructed, and outside of
        those conditions the format resolves only to ciphertext and synchronized
        noise.
        <br />
        <br />
        improvements
        <br />
        scale + speed. scale + speeeeeedd. <br />
        <br />
        1. handle longer videos
        <br />
        2. better sam3 masking and real time not pre-processed
        <br />
        3. the current 1.3b lora works, but it loses a lot of fine detail
        compared to its 14b counterpart
        <br />
        <br />
        all of this ultimately comes down to memory, ram, gpus etc
        <br />
        <br />
        why y2k?
        <br />
        because as everything gets endshitified, y2k is the style that still
        remembers web1, when things felt possible and not drowned in polarizing
        bot slop (nothing against bots, everything against their overlords)
        <br />
        <br />
        <br />
        <br />
        privacy is a right.
        <br />
        most people don't care to get that.
        <br />
        why wait. why ask. why depend.
        <br />
        <br />
        reshape how you come into focus.
        <br />
        edit your mediums.
      </CardContent>
    </Card>
  );
}
