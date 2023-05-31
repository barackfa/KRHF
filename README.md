# Kognitív robotika Házi Feladat
Kognitív robotika HF1

Feladatunk vobalkövetés megvalósítása volt turtlebottal és neurűlis hálóval.
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



### Telepítési útmutató
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


## Felhasznált csomagok
http://wiki.ros.org/Robots/TurtleBot#Robots.2FTurtleBot.2Fkinetic.Packages

http://wiki.ros.org/rviz

http://wiki.ros.org/gazebo_ros_pkgs



## Videó

A képre való kattintást követően továbbirányít.
[![videó link](https://i3.ytimg.com/vi/cpri8pE5wGg/maxresdefault.jpg)](https://youtu.be/cpri8pE5wGg)



# ettől
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

Kellő mennyiségű kép készítését követően osztályozni kell, majd pedig tanítani a modellt. Ezt a számítógépen végezzük, mert a robot tárhely/RAM kapacitása kisebb egy személyi számítógéphez képest kevés. Emiatt a ```line_follower_cnn.py``` fájlt is módosítani kell tensorflow lite csomagra, a modellt is át kell alakítani.
# eddig újragondolni

Dalalalala
Dalalalala






