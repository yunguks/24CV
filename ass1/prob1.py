import cv2 
import random
import numpy as np 
import matplotlib.pyplot as plt
import os
import argparse

COLOR = [
    [0, 'Green', (0,255,0)], 
    [1, 'Blue', (0,0,255)], 
    [2, 'Red', (255,0,0)], 
    [3, 'Yellow' , (255, 255, 0)], 
    [4, "Purple", (128,0,128)], 
    [5, 'Orange',(255,165,0)] 
    ]

IMG_SHAPE = (227,227,3)
CENTER = 227//2


def get_args():
    parser = argparse.ArgumentParser(description="Make clock dataset")
    # data
    parser.add_argument("--dataset", choices=['all', 'sample'], default ='sample', help="Choose how much you want to make data. Default is sample which make only one data.")
    
    # iter
    parser.add_argument('--iter', type=int, default=1, help='Choose how many times you want to repeat it. Default is 1.')
    
    # save
    parser.add_argument('--save_folder', type=str, default='./train', help='Choose where save data you want to. Default is "./train".')
    
    
    return parser.parse_args()


def make_img(shape, color, apm, hour, minute):
    # 빈 배경 만들기
    img = np.ones(IMG_SHAPE) * 127

    # 0 = circle, 1 = rectangle
    if shape == 'rectangle':
        img = cv2.rectangle(img, (0,0), (227,227), color[-1], -1)
    else:
        img = cv2.circle(img, (CENTER,CENTER), 113, color[-1], -1)


    length = 90

    # minute, 분침 만들기
    minute_angle = 6*(minute-15)
    minute_end = (int(CENTER+(length*(np.cos(np.radians(minute_angle))))), CENTER+int(length*(np.sin(np.radians(minute_angle)))))
    cv2.line(img, (CENTER, CENTER), minute_end, (0, 0, 0), thickness=3)

    # hour, 시침 만들기
    hour_angle = 30*(hour-3)+(minute//2)
    hour_end = (int(CENTER+length*(np.cos(np.radians(hour_angle)))), int(CENTER+length*(np.sin(np.radians(hour_angle)))) )
    cv2.line(img, (CENTER, CENTER), hour_end, (0, 0, 0), thickness=5)

    # DIGIT 그리기
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    font_thickness = 2
    for i in range(1, 13):
        angle = (i+3) * 30  # Calculate angle for each digit
        digit = str(i)
        text_size = cv2.getTextSize(digit, font, font_scale, font_thickness)[0]
        text_origin = (int(CENTER - 100 * np.cos(np.radians(angle))) - text_size[0]//2,
                    int(CENTER - 100 * np.sin(np.radians(angle))) + text_size[1]//2)
        cv2.putText(img, digit, text_origin, font, font_scale, (0, 0, 0), font_thickness, cv2.LINE_AA)
    
    # AM PM 그림 그리기
    cv2.putText(img , apm, (10,20), font, font_scale, (0, 0, 0), font_thickness, cv2.LINE_AA)

    # noise 추가
    noise = np.random.normal(0, 80, img.shape)
    img = img + noise
    img = np.clip(img, 0, 255)
    img = img.astype(np.uint8)
    
    return img

    

if __name__ == "__main__":
    
    args = get_args()
    
    
    # image 저장 폴더 만들기
    save_foler = args.save_folder
    if not os.path.exists(save_foler):
        os.mkdir(save_foler)
        print(f"create {save_foler}")
    
    
    for i in range(args.iter):
        
        # 하나의 sample만 만들기
        if args.dataset == 'sample':
            minutes = [random.randint(0, 59)]
            hours = [random.randint(0, 11)]
            apms = [random.choice(['AM', 'PM'])]
            clock_shapes = [random.choice(['circle', 'rectangle'])]
            colors = [random.choice(COLOR)]
        
        # 가능한 전체 조합 만들기
        else:
            minutes = list(range(0,60))
            hours = list(range(0,12))
            apms = ['AM', 'PM']
            clock_shapes = ['circle', 'rectangle']
            colors = [color for color in COLOR]
            
        print(f"Making {i+1}th dataset....")
        for shape in clock_shapes:
            for color in colors:
                for apm in apms:
                    for hour in hours:
                        for minute in minutes:
                            
                            img = make_img(shape, color, apm, hour, minute)
                            count = 0

                            while True:
                                # 파일이름을 circle_red_ap_10_15_count.png 정답을 알 수 있게 저장
                                # count는 동일한 이름이 있는 경우 몇개인지
                                name = f"{shape}_{color[1]}_{apm}_{hour}_{minute}_{count}.png"
                                save_path = os.path.join(save_foler, name)
                                if os.path.exists(save_path):
                                    count+=1
                                else:
                                    break
                                
                                
                            cv2.imwrite(save_path, cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
