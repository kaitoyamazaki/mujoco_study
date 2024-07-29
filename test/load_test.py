import mujoco

xml = """
<mujoco>
  <worldbody>
    <geom name="red_box" type="box" size=".2 .2 .2" rgba="1 0 0 1"/>
    <geom name="green_sphere" pos=".2 .2 .2" size=".1" rgba="0 1 0 1"/>
  </worldbody>
</mujoco>
"""
model = mujoco.MjModel.from_xml_string(xml)

## XMLファイルとして書き出す
#mujoco.mj_saveLastXML("arm.xml", model)

print(f'{model.ngeom}')
print(f'{model.geom_rgba}')

print(f'{model.geom("red_box").id}')
print(f'{model.geom(0).name}')

print(f'{model.nbody}')  
print(f'{model.body("world").id}')

data = mujoco.MjData(model)

print(f'{data.time}, {data.qpos}, {data.qvel}')
#print(data.time, data.qpos, data.qvel)

print(f'{data.geom_xpos}')

mujoco.mj_kinematics(model, data)
print(data.geom(1).xpos)