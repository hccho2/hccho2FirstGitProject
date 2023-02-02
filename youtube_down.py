'''
cmd에서도 가능
> youtube-dl -x --audio-format mp3 https://www.youtube.com/watch?v=4J5cKU5IUAI&t=4678s
> youtube-dl -f bestaudio https://www.youtube.com/watch?v=4J5cKU5IUAI&t=4678s --output "out.%(ext)s"
> youtube-dl -x --audio-format mp3 --audio-quality 0 https://www.youtube.com/watch?v=4J5cKU5IUAI&t=4678s



't' is not recognized as an internal or external command, operable program or batch file.   --> 관리자권한으로 실행해야 한다.

'''


import os
import youtube_dl # pip install youtube-dl

VIDEO_DOWNLOAD_PATH = './'  # 다운로드 경로

def download_video_and_subtitle(output_dir, youtube_video_list):

    download_path = os.path.join(output_dir, '%(id)s-%(title)s.%(ext)s')

    for video_url in youtube_video_list:

        # youtube_dl options
        ydl_opts = {
            'format': 'best/best',  # 가장 좋은 화질로 선택(화질을 선택하여 다운로드 가능)
            'outtmpl': download_path, # 다운로드 경로 설정
            'writesubtitles': 'best', # 자막 다운로드(자막이 없는 경우 다운로드 X)
            'writethumbnail': 'best',  # 영상 thumbnail 다운로드
            'writeautomaticsub': True, # 자동 생성된 자막 다운로드
            'subtitleslangs': 'en'  # 자막 언어가 영어인 경우(다른 언어로 변경 가능)
        }

        try:
            with youtube_dl.YoutubeDL(ydl_opts) as ydl:
                ydl.download([video_url])
        except Exception as e:
            print('error', e)

def download():
    # short https://youtube.com/shorts/nca6easwPB4  ---> https://youtube.com/watch?v=nca6easwPB4
    youtube_url_list = [ 'https://youtube.com/watch?v=nca6easwPB4' ]
    download_video_and_subtitle(VIDEO_DOWNLOAD_PATH, youtube_url_list)

def convert():
    import moviepy.editor as mp

    clip = mp.VideoFileClip("이문세BEST.mp4")
    clip.audio.write_audiofile("이문세BEST.mp3")
if __name__ == '__main__':
    download()
    #convert()

    print('Complete download!')