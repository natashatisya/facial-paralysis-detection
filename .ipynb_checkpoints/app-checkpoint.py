{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "0fe68d55",
   "metadata": {},
   "outputs": [],
   "source": [
    "from flask import Flask, render_template, request\n",
    "from tensorflow.keras.models import load_model\n",
    "import numpy as np\n",
    "from tensorflow.keras.preprocessing import image"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "id": "609c5732",
   "metadata": {},
   "outputs": [],
   "source": [
    "app = Flask(__name__, template_folder='.', static_folder='static')\n",
    "\n",
    "# Load the TensorFlow model\n",
    "model = load_model('C:/Users/Natasha Tisya/Documents/DEGREE/SEM 6 DEGREE/CSP650/SYSTEM FACIAL PARALYSIS/Coding System FYP Project/Model/inception_resnetv2_model.h5')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "id": "4da5ee17",
   "metadata": {},
   "outputs": [
    {
     "ename": "AssertionError",
     "evalue": "View function mapping is overwriting an existing endpoint function: home",
     "output_type": "error",
     "traceback": [
      "\u001b[1;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[1;31mAssertionError\u001b[0m                            Traceback (most recent call last)",
      "Cell \u001b[1;32mIn[6], line 11\u001b[0m\n\u001b[0;32m      7\u001b[0m     result \u001b[38;5;241m=\u001b[39m model\u001b[38;5;241m.\u001b[39mpredict(img)\n\u001b[0;32m      8\u001b[0m     \u001b[38;5;28;01mreturn\u001b[39;00m result\n\u001b[0;32m     10\u001b[0m \u001b[38;5;129;43m@app\u001b[39;49m\u001b[38;5;241;43m.\u001b[39;49m\u001b[43mroute\u001b[49m\u001b[43m(\u001b[49m\u001b[38;5;124;43m'\u001b[39;49m\u001b[38;5;124;43m/\u001b[39;49m\u001b[38;5;124;43m'\u001b[39;49m\u001b[43m)\u001b[49m\n\u001b[1;32m---> 11\u001b[0m \u001b[38;5;28;43;01mdef\u001b[39;49;00m\u001b[43m \u001b[49m\u001b[38;5;21;43mhome\u001b[39;49m\u001b[43m(\u001b[49m\u001b[43m)\u001b[49m\u001b[43m:\u001b[49m\n\u001b[0;32m     12\u001b[0m \u001b[43m    \u001b[49m\u001b[38;5;28;43;01mreturn\u001b[39;49;00m\u001b[43m \u001b[49m\u001b[43mrender_template\u001b[49m\u001b[43m(\u001b[49m\u001b[38;5;124;43m'\u001b[39;49m\u001b[38;5;124;43mindex.html\u001b[39;49m\u001b[38;5;124;43m'\u001b[39;49m\u001b[43m)\u001b[49m\n\u001b[0;32m     14\u001b[0m \u001b[38;5;129m@app\u001b[39m\u001b[38;5;241m.\u001b[39mroute(\u001b[38;5;124m'\u001b[39m\u001b[38;5;124m/upload\u001b[39m\u001b[38;5;124m'\u001b[39m)\n\u001b[0;32m     15\u001b[0m \u001b[38;5;28;01mdef\u001b[39;00m \u001b[38;5;21mupload\u001b[39m():\n",
      "File \u001b[1;32m~\\anaconda3\\lib\\site-packages\\flask\\scaffold.py:449\u001b[0m, in \u001b[0;36mScaffold.route.<locals>.decorator\u001b[1;34m(f)\u001b[0m\n\u001b[0;32m    447\u001b[0m \u001b[38;5;28;01mdef\u001b[39;00m \u001b[38;5;21mdecorator\u001b[39m(f: T_route) \u001b[38;5;241m-\u001b[39m\u001b[38;5;241m>\u001b[39m T_route:\n\u001b[0;32m    448\u001b[0m     endpoint \u001b[38;5;241m=\u001b[39m options\u001b[38;5;241m.\u001b[39mpop(\u001b[38;5;124m\"\u001b[39m\u001b[38;5;124mendpoint\u001b[39m\u001b[38;5;124m\"\u001b[39m, \u001b[38;5;28;01mNone\u001b[39;00m)\n\u001b[1;32m--> 449\u001b[0m     \u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39madd_url_rule(rule, endpoint, f, \u001b[38;5;241m*\u001b[39m\u001b[38;5;241m*\u001b[39moptions)\n\u001b[0;32m    450\u001b[0m     \u001b[38;5;28;01mreturn\u001b[39;00m f\n",
      "File \u001b[1;32m~\\anaconda3\\lib\\site-packages\\flask\\scaffold.py:50\u001b[0m, in \u001b[0;36msetupmethod.<locals>.wrapper_func\u001b[1;34m(self, *args, **kwargs)\u001b[0m\n\u001b[0;32m     48\u001b[0m \u001b[38;5;28;01mdef\u001b[39;00m \u001b[38;5;21mwrapper_func\u001b[39m(\u001b[38;5;28mself\u001b[39m, \u001b[38;5;241m*\u001b[39margs: t\u001b[38;5;241m.\u001b[39mAny, \u001b[38;5;241m*\u001b[39m\u001b[38;5;241m*\u001b[39mkwargs: t\u001b[38;5;241m.\u001b[39mAny) \u001b[38;5;241m-\u001b[39m\u001b[38;5;241m>\u001b[39m t\u001b[38;5;241m.\u001b[39mAny:\n\u001b[0;32m     49\u001b[0m     \u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39m_check_setup_finished(f_name)\n\u001b[1;32m---> 50\u001b[0m     \u001b[38;5;28;01mreturn\u001b[39;00m f(\u001b[38;5;28mself\u001b[39m, \u001b[38;5;241m*\u001b[39margs, \u001b[38;5;241m*\u001b[39m\u001b[38;5;241m*\u001b[39mkwargs)\n",
      "File \u001b[1;32m~\\anaconda3\\lib\\site-packages\\flask\\app.py:1358\u001b[0m, in \u001b[0;36mFlask.add_url_rule\u001b[1;34m(self, rule, endpoint, view_func, provide_automatic_options, **options)\u001b[0m\n\u001b[0;32m   1356\u001b[0m old_func \u001b[38;5;241m=\u001b[39m \u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39mview_functions\u001b[38;5;241m.\u001b[39mget(endpoint)\n\u001b[0;32m   1357\u001b[0m \u001b[38;5;28;01mif\u001b[39;00m old_func \u001b[38;5;129;01mis\u001b[39;00m \u001b[38;5;129;01mnot\u001b[39;00m \u001b[38;5;28;01mNone\u001b[39;00m \u001b[38;5;129;01mand\u001b[39;00m old_func \u001b[38;5;241m!=\u001b[39m view_func:\n\u001b[1;32m-> 1358\u001b[0m     \u001b[38;5;28;01mraise\u001b[39;00m \u001b[38;5;167;01mAssertionError\u001b[39;00m(\n\u001b[0;32m   1359\u001b[0m         \u001b[38;5;124m\"\u001b[39m\u001b[38;5;124mView function mapping is overwriting an existing\u001b[39m\u001b[38;5;124m\"\u001b[39m\n\u001b[0;32m   1360\u001b[0m         \u001b[38;5;124mf\u001b[39m\u001b[38;5;124m\"\u001b[39m\u001b[38;5;124m endpoint function: \u001b[39m\u001b[38;5;132;01m{\u001b[39;00mendpoint\u001b[38;5;132;01m}\u001b[39;00m\u001b[38;5;124m\"\u001b[39m\n\u001b[0;32m   1361\u001b[0m     )\n\u001b[0;32m   1362\u001b[0m \u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39mview_functions[endpoint] \u001b[38;5;241m=\u001b[39m view_func\n",
      "\u001b[1;31mAssertionError\u001b[0m: View function mapping is overwriting an existing endpoint function: home"
     ]
    }
   ],
   "source": [
    "# Function to make predictions for a single image\n",
    "def predict_single_image(image_path, model):\n",
    "    img = image.load_img(image_path, target_size=(224, 224))\n",
    "    img = image.img_to_array(img)\n",
    "    img = img / 255.0  # Normalize the image\n",
    "    img = np.expand_dims(img, axis=0)  # Add batch dimension\n",
    "    result = model.predict(img)\n",
    "    return result\n",
    "\n",
    "@app.route('/')\n",
    "def home():\n",
    "    return render_template('index.html')\n",
    "\n",
    "@app.route('/upload')\n",
    "def upload():\n",
    "    return render_template('upload.html')\n",
    "\n",
    "@app.route('/preview')\n",
    "def preview():\n",
    "    return render_template('previewimage.html')\n",
    "\n",
    "@app.route('/predict', methods=['POST'])\n",
    "def predict():\n",
    "    if 'image' in request.files:\n",
    "        image_file = request.files['image']\n",
    "        if image_file.filename != '':\n",
    "            image_path = 'path_to_save_uploaded_image'  # Set the path to save the uploaded image\n",
    "            image_file.save(image_path)\n",
    "            result = predict_single_image(image_path, model)\n",
    "            if result > 0.5:\n",
    "                classification_result = \"Stroke Face\"\n",
    "            else:\n",
    "                classification_result = \"Normal Face\"\n",
    "            return render_template('prediction.html', classification_result=classification_result)\n",
    "\n",
    "    return render_template('predict.html')\n",
    "\n",
    "if __name__ == '__main__':\n",
    "    app.run(debug=True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "06c5f040",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.10.9"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
