# Kognitív robotika Házi Feladat

Feladatunk vonalkövetés megvalósítása volt turtlebottal és neurűlis hálóval.
A neurális hálót a roboton kellett futtatnunk egy külön számítógép helyett.
Ennek érdekében minél kisebb és gyorsabb, de a feladat megvalósításához még megfelelő méretű neurális háló létrehozására törekedtünk.
Az elkészült vonalkövető rendszert a valós roboton kellett tesztelnünk és bemutatnunk.

A feladat megvalósítását a kis méretű neurális háló létrehozásával kezdtük. 
Az elkészült neurális hálót kezdetben a szimulációs környezetben teszteltük a gyakorlatokon tanultaknak megfelelően.
A robot a szimulációs környezetben megfelelően működött, követte a vonalat.
Ezután áttértünk a valós roboton való tesztelésre. 
Ekkor több problémába is ütköztünk. A robottal való kommunikációs problémák megoldása után kiderült, hogy a Tensorflow könyvtárat nem tudjuk telepíteni a robotra,
méretproblémák miatt. Így a Tensorflow Lite verziót kellett alkalmaznunk, amihez a már elkészített modellt is át kellett alakítanunk ennek megfelelően.
Ezzel párhuzamosan leteszteltük a neurális háló működését, ami a szimulációs környezethez képest sokkal rosszabb eredményeket mutatott.
A valós robottal több, mint 100 képet készítettünk a valós környezetről, annak érdekében, hogy a neurális hálónk jobban tudjon alkalmazkodni a valós helyzethez.
Így egy újabb tanítás és neurális háló egyszerűsítés következett.
Az új, elkészült modellt átalakítottuk a Tensorflow Lite könyvtárnak megfelelően, és ismét szimulációban teszteltük a működését.
Ezután a valós roboton is leteszteltük az elkészült vonalkövető rendszert.


## Tartalomjegyzék

1. [Telepítési útmutató](#telepítési-útmutató)
2. [Dependency-k](#dependency-k)
3. [Felhasznált csomagok](#felhasznált-csomagok)
4. [Videó](#videó)
5. [A neurális háló tanítása](#a-neurális-háló-tanítása)
    - [Modell létrehozása](#modell-létrehozása)
    - [A modell átkonvertálása](#a-modell-átkonvertálása)
    - [Új line follower node létrehozása](#új-line-follower-node-létrehozása)
    - [A megtanított modell tesztelése](#a-megtanított-modell-tesztelése)



## Telepítési útmutató
Első lépésként hozzuk létre a csomagunkat tartalmazó workspace-t. Majd a catkin_make paranccsal fordítsuk.
```
cd ~
mkdir -p catkin_ws/src
cd catkin_ws
catkin_make
```
Ezt követően sorce-olható a workspace.
```
source ~/catkin_ws/devel/setup.bash
```
Ezt betehetjük a .bashrc fájlba is.
```
cd ~
nano .bashrc
--------------------------------------------------
WORKSPACE=~/catkin_ws/devel/setup.bash
source $WORKSPACE
```
Ezt követően töltsük le a MOGI turtlebot3 módosított verzióját.
```
cd catkin_ws/src/
git clone https://github.com/barackfa/KRHF
cd catkin_ws/ 
catkin_make
```


A robotra való bejeltkezést ssh-val oldjuk meg. Ehhez szükséges, hogy azonos hálón legyünk a robottal. A laborban lévő 2 robot ip címe: ```192.168.50.111``` valamint ```192.168.50.175```, ezek közül az egyikre lesz szükségünk.
```
ssh 192.168.50.111 -lubuntu
vagy
ssh 192.168.50.175 -lubuntu
pw: ubuntu
```
A turtlebot véges memória kapacitása miatt tensorflow lite-ot kell telepíteni, és ezzel futtatható a tanított és aztán átkonvertált neurális háló. 
```
python3 -m pip install tflite-runtime
```
Ezt követően át kell mozgatni a csomagot a robotra, ezt megtehetjük parancsból és ubuntu rendszer alatt a fájlrendszerben a robotra csatlakozva is.
```
scp -r /path/to/local/source user@ssh.example.com:/path/to/remote/destination 
```

## Dependency-k

![Kép link](https://github.com/barackfa/KRHF/blob/main/assets/graph.png)


## Felhasznált csomagok
http://wiki.ros.org/Robots/TurtleBot#Robots.2FTurtleBot.2Fkinetic.Packages

http://wiki.ros.org/rviz

http://wiki.ros.org/gazebo_ros_pkgs

http://wiki.ros.org/rqt_dep



## Videó

A képre való kattintást követően továbbirányít.
[![videó link](https://i3.ytimg.com/vi/cpri8pE5wGg/maxresdefault.jpg)](https://youtu.be/cpri8pE5wGg)



## A neurális háló tanítása

### Modell létrehozása
A neurális háló futtatásához, először tanítani kell ezt. Ehhez képeket készítettünk, majd ezeket osztályoztuk és tanítottuk a CNN-t.
A képek készítéshez először indítsuk el a roboton a bringup fájlt.
```
roslaunch turtlebot3_bringup turtlebot3_robot.launch
```
Ezt követően indítsunk egy másik terminált, amelyben állítsuk be ROS Mastert, ezt az ssh után kapott üzenetből tudjuk kimásolni.
```
export ROS_MASTER_URI=http://192.168.1.111:11311
```
A képek készítését a ```save_training_images.py``` végzi el.

```
rosrun turtlebot3_mogi save_training_images.py
```
Amennyiben egyedül vagyunk, akkor egy újabb terminálban a teleop indításával megoldható, hogy a robotot irányítsuk, átmozgasssuk.
```
roslaunch turtlebot3_teleop turtlebot3_teleop_key.launch
```
Hogy minél több képpel tudjuk elvégezni a modell betanítását, és hogy kellően generikus legyen az, a szimulációban futtatva is készítettünk több képet.
Ezeket a képeket pedig osztályoztuk a laboron látott módon annak megfelelően, hogy mit szeretnénk, hogy csináljon a robot, mikor az adott képet kapja bemenetként.
Majd ezekkel a képekkel végezzük a tanítást a ```train_network.py``` segítségével számítógépen, mert a robot tárhely/RAM kapacitása kisebb egy személyi számítógépéhez képest.

Egyik fő kihívás a feladatunkban, hogy kisebb méretű neurális hálót állítsunk elő, ami még működő képes. Ehhez módosítanunk kellett az órán kiadott ```train_network.py``` kódot.
Több lehetőséggel is próbálkoztunk, például a 3 rétegű RGB képek 1 rétegű szürkeárnyalatossá alakításával, azonban ez nem hozta meg a kívánt hatást.
Az így készített neurális háló mérete nem csökkent számottevően, és kisebb pontossággal tudott betanulni.
A megoldást végül az jelentette, hogy a tanuláshoz használt kép méretét levettük mindössze 8x8-asra, illetve a LeNet-5 jellegű hálónál csökkentettük a paramétereket, kisebbra szabtuk a robotunk "agyát".

```python

    # first set of CONV => RELU => POOL layers
    model.add(Conv2D(6, (5, 5), padding="same", input_shape=inputShape))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    # second set of CONV => RELU => POOL layers
    model.add(Conv2D(20, (5, 5), padding="same"))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    # first (and only) set of FC => RELU layers
    model.add(Flatten())
    model.add(Dense(500))
    model.add(Activation("relu"))
```
Így már kellő pontossággal sikerült betanítanunk a modellünket, ami kb. huszad akkora lett, mint a laboron készített.

![Kép link](https://github.com/barackfa/KRHF/blob/main/assets/model_training.png)

### A modell átkonvertálása
Mivel TensorFlow helyett csak a TensorFlow Lite verziót tudjuk használni, ezért szükséges a modellt átkonvertálnunk. Ehhez létrehoztunk egy külön lefuttatható Python scriptet, amely segítségével a korábban generált Keras modellből TFLite modellt készítünk a megfelelő path-ok megadásával:

```python
import tensorflow as tf

# Load the original Keras model
model = tf.keras.models.load_model('/home/catkin_ws/src/turtlebot3/turtlebot3_mogi/network_model/model.best.h5')

# Convert the model to TensorFlow Lite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save the TFLite model.
with open('/home/catkin_ws/src/turtlebot3/turtlebot3_mogi/network_model/model.tflite', 'wb') as f:
    f.write(tflite_model)
```

### Új line follower node létrehozása
A vonalkövetés megvalósítását a TensorFlow Lite csomag segítségével végezzük. Emiatt az órán megírt ```line_follower_cnn.py``` fájt több helyen is módosítanunk kellett.

A több, TensorFlow-hoz szükséges csomag helyett elegendő a TFLite-ot importálni:
```python
import tflite_runtime.interpreter as tflite
```

TensorFlow esetén elég volt betölteni a modellt a ```load_model``` segítségével, a predikció pedig a ```model(image)``` segítségével történt. Azonban TensorFLow Lite használata esetén létre kellett hozni egy ```Interpreter``` objektumot, allokálni a tenzorokat, majd a predikcióhoz használni a ```set_tensor``` és ```invoke``` függvényeket.

```python
# Load TFLite model and allocate tensors
interpreter = tflite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
```

A képfeldolgozó függvény módosítása is szükséges volt. A képet array-é alakítás után 4D-ssé formáztuk, mivel a TensorFlow Lite modellnek szüksége van a batch dimenzió megadására is a futáshoz.
```python
def processImage(self, img):
        image = cv2.resize(img, (image_size, image_size))
        image = np.reshape(image, (-1, image_size, image_size, 3)).astype("float32") / 255.0

        # Set the value of the input tensor
        interpreter.set_tensor(input_details[0]['index'], image)
        interpreter.invoke()

        # Retrieve the value of the output tensor
        prediction = np.argmax(interpreter.get_tensor(output_details[0]['index']))
        
        [...]
```

Ezek a legfőbb módosítások, a többi (pl. verziók kiíratása és ellenőrzése, megjelenítéssel kapcsolatos funkciók eltávolítása) megtalálható a feltöltött ```line_follower_cnn_hf.py``` fájlban.

### A megtanított modell tesztelése

Az elkészített modellt először szimulációban számítógépen futtatva teszteltük. Itt több szenárióba elhelyezve is meggyőzödhettünk a modell helyes működésén.
![Kép link](https://github.com/barackfa/KRHF/blob/main/assets/kep01.png)

Végül pedig a laborban a roboton futtatva is teszteltük a modellt, ami sikeresen követte a vonalat.
![Kép link](https://github.com/barackfa/KRHF/blob/main/assets/kep02.jpeg)
