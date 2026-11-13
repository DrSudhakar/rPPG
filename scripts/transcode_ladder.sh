#!/usr/bin/env bash
# Make the SAMA compression ladder from one uncompressed (or lossless) video.
# Usage: ./transcode_ladder.sh input.avi outdir
# Output: lossless.mkv (FFV1), h264_{2000,1000,500,250}k.mp4, vp8_{500,250}k.webm, all 720p max, 30 fps.
# Requires ffmpeg with libx264 and libvpx.
set -euo pipefail
in="$1"; out="${2:-ladder}"
mkdir -p "$out"
scale="scale='min(1280,iw)':-2,fps=30"

# lossless reference copy (keeps every pulse detail)
ffmpeg -y -loglevel error -i "$in" -vf "$scale" -c:v ffv1 -level 3 -pix_fmt yuv444p "$out/lossless.mkv"

# H.264 at fixed bitrates, two-pass, like a video call's constant bitrate
for kb in 2000 1000 500 250; do
  ffmpeg -y -loglevel error -i "$in" -vf "$scale" -c:v libx264 -preset veryfast -b:v ${kb}k -maxrate ${kb}k \
         -bufsize $((kb*2))k -pix_fmt yuv420p -pass 1 -an -f mp4 /dev/null
  ffmpeg -y -loglevel error -i "$in" -vf "$scale" -c:v libx264 -preset veryfast -b:v ${kb}k -maxrate ${kb}k \
         -bufsize $((kb*2))k -pix_fmt yuv420p -pass 2 -an "$out/h264_${kb}k.mp4"
done

# VP8, the codec WebRTC video calls must support
for kb in 500 250; do
  ffmpeg -y -loglevel error -i "$in" -vf "$scale" -c:v libvpx -b:v ${kb}k -minrate ${kb}k -maxrate ${kb}k \
         -deadline realtime -cpu-used 8 -pix_fmt yuv420p -an "$out/vp8_${kb}k.webm"
done
rm -f ffmpeg2pass-0.log ffmpeg2pass-0.log.mbtree
echo "ladder written to $out"
