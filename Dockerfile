#docker build -t kult:py310 .
#docker run -it --gpus all -v `pwd`:/app -p 8080:8080 -p 5000:5000/udp --name kult kult:py310 bash
#jupyter lab --ip=0.0.0.0 --no-browser --port=8080 --allow-root

FROM python:3.10

ENV DEBIAN_FRONTEND=noninteractive
RUN apt update && apt install -y locales locales-all
ENV LANG en_US.UTF-8
ENV LANGUAGE en_US:en
ENV LC_ALL en_US.UTF-8

RUN apt update
RUN apt install -y --no-install-recommends python3-pip python3-dev nano wget git git-lfs ffmpeg libsm6 libxext6
RUN python3 -m pip install --no-cache-dir setuptools
RUN pip install --upgrade pip

#video-face
RUN pip install cmake
RUN pip install opencv-python pillow tf-keras==2.17 dlib deepface

#audio-speech train vera
RUN pip install matplotlib pandas librosa plotly seaborn antropy np_utils pydub kagglehub fastparquet
RUN pip install torch torchvision

#audio-text
RUN pip install openai-whisper transformers

#server
RUN apt install -y autoconf automake nasm build-essential cmake git libtool pkg-config texinfo wget yasm libssl-dev libcurl4-openssl-dev
RUN mkdir /ffmpeg_sources
WORKDIR /ffmpeg_sources
RUN git clone https://github.com/Haivision/srt.git
WORKDIR /ffmpeg_sources/srt
RUN ./configure
RUN make -j$(nproc)
RUN make install
WORKDIR /ffmpeg_sources
RUN wget https://ffmpeg.org/releases/ffmpeg-snapshot.tar.bz2
RUN tar xjf ffmpeg-snapshot.tar.bz2
WORKDIR /ffmpeg_sources/ffmpeg
RUN PKG_CONFIG_PATH="/usr/local/lib/pkgconfig" ./configure \
--prefix="/usr/local" \
--extra-cflags="-I/usr/local/include" \
--extra-ldflags="-L/usr/local/lib" \
--bindir="/usr/local/bin" \
--enable-libsrt \
--enable-gpl \
--enable-nonfree \
--enable-shared \
--disable-static \
--disable-debug \
--disable-doc \
--disable-ffplay
RUN make -j$(nproc)
RUN make install
RUN ldconfig
RUN PKG_CONFIG_PATH=/usr/local/lib/pkgconfig pip install av --no-binary av

#RUN pip install datasets
#RUN pip install torch torchvision mediapipe scikit-learn transformers sentence-transformers gensim protobuf
RUN pip install jupyter

WORKDIR /app

RUN pip install -e .

EXPOSE 8080
EXPOSE 5000

#RUN git clone https://github.com/CheyneyComputerScience/CREMA-D.git