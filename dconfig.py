import os


class Config(object):
    pass


class ProductionConfig(Config):
    DEBUG = False
    SERVER_NAME = os.getenv('SERVER_NAME')
    PORT_NUMBER = os.getenv('PORT_NUMBER')

    VND_ID_INTERNAL_DOMAIN = os.getenv('VND_ID_INTERNAL_DOMAIN')
    SECRET_AUTHEN_KEY = os.getenv('SECRET_AUTHEN_KEY')

    DATA_DIR = os.getenv('DATA_DIR')
    LOG_DIR = os.getenv('LOG_DIR')
    WEB_STATIC_DIR = os.getenv('WEB_STATIC_DIR')

    MODEL_FACE_ANALYSIS_NAME = os.getenv('MODEL_FACE_ANALYSIS_NAME')
    MODEL_MASK_DIR = os.getenv('MODEL_MASK_DIR')
    YOLOV5_DIR = os.getenv('YOLOV5_DIR')


class DevelopmentConfig(Config):
    if os.getenv('DEBUG'):
        DEBUG = os.getenv('DEBUG') == 'True'
    else:
        DEBUG = True
    if os.getenv('SERVER_NAME'):
        SERVER_NAME = os.getenv('SERVER_NAME')
    else:
        SERVER_NAME = "0.0.0.0"

    if os.getenv('PORT_NUMBER'):
        PORT_NUMBER = os.getenv('PORT_NUMBER')
    else:
        PORT_NUMBER = 5000

    VND_ID_INTERNAL_DOMAIN = 'https://accounts-uat.vndirect.com.vn/'
    if os.getenv('SECRET_AUTHEN_KEY'):
        SECRET_AUTHEN_KEY = os.getenv('SECRET_AUTHEN_KEY')
    else:
        SECRET_AUTHEN_KEY = 'tX5YKLeoCEBDYehSUjSDrFAqqbPIwpO3h52A1ZI7hBPjJWrX4zkMEZLAQDOD'

    if os.getenv('CDN_SERVER'):
        CDN_SERVER = os.getenv('CDN_SERVER')
    else:
        CDN_SERVER = 'https://face-matching-cdn.vndirect.com.vn/'

    if os.getenv('CND_FOLDER_PATH'):
        CND_FOLDER_PATH = os.getenv('CDN_SERVER')
    else:
        CND_FOLDER_PATH = 'ai_face_matching'

    if os.getenv('VOC_API'):
        VOC_API = os.getenv('VOC_API')
    else:
        VOC_API = 'https://tttv-api-suat.vndirect.com.vn/'

    if os.getenv('DATA_DIR'):
        DATA_DIR = os.getenv('DATA_DIR')
    else:
        DATA_DIR = '/opt/gitlab/vnd-face-matching-api/data'
    if os.getenv('LOG_DIR'):
        LOG_DIR = os.getenv('LOG_DIR')
    else:
        LOG_DIR = '/opt/gitlab/vnd-face-matching-api/log'
    if os.getenv('WEB_STATIC_DIR'):
        WEB_STATIC_DIR = os.getenv('WEB_STATIC_DIR')
    else:
        WEB_STATIC_DIR = '/opt/gitlab/vnd-face-matching-api/app/web'

    if os.getenv('DEVICE'):
        DEVICE = os.getenv('DEVICE')
    else:
        DEVICE = 'CPU'

    if os.getenv('MODEL_FACE_ANALYSIS_NAME'):
        MODEL_FACE_ANALYSIS_NAME = os.getenv('MODEL_FACE_ANALYSIS_NAME')
    else:
        MODEL_FACE_ANALYSIS_NAME = 'antelopev2'

    if os.getenv('MODEL_MASK_DIR'):
        MODEL_MASK_DIR = os.getenv('MODEL_MASK_DIR')
    else:
        MODEL_MASK_DIR = '/opt/gitlab/vnd-face-matching-api/app/retinaface_anticov/cov2/mnet_cov2'

    if os.getenv('YOLOV5_DIR'):
        YOLOV5_DIR = os.getenv('YOLOV5_DIR')
    else:
        YOLOV5_DIR = '/opt/gitlab/vnd-face-matching-api/app/yolov5'


config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
}

VERSION = "1.0.5"

config_object = config[os.getenv('FLASK_ENV') or 'development']
