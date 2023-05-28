# KRHF
Kognitív robotika HF

Feladatunk kisméretű neurális hálóval vonalat követni valós roboton.


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
Ezt követően töltsük le a MOGI turtlebot3 verziót, majd a ```mogi-ros``` branch-re lépjünk át.
```
git clone https://github.com/MOGI-ROS/turtlebot3
git checkout mogi-ros
````
?? itt kell catkin_ws??

Ezt követően töltsük le a line_follower_cnn.py és model.best.h5 fájlokat és cseréljük le a mogi turtlebot3 meglévő fájljait.

A robotra való bejeltkezést ssh-val oldjuk meg. Ehhez szükséges, hogy azonos hálón legyünk a robottal. A laborban lévő 2 robot ip címe: ```192.168.50.111``` valamint ```192.168.50.175```, ezek közül az egyikre lesz szükségünk.
```
ssh 192.168.50.111 -lubuntu
vagy
ssh 192.168.50.175 -lubuntu
pw: ubuntu
```

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









