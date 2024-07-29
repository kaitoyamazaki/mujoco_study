import mujoco

xml = """
<mujoco>
  <default class="main">
    <geom/>
  </default>

  <worldbody>
    <geom/>
    <body name="b1">
      <geom/>
      <body name="b3">
        <joint/>
        <geom/>
      </body>
    </body>
    <body name="b2">
      <joint/>
      <geom/>
    </body>
  </worldbody>
</mujoco>
"""
model = mujoco.MjModel.from_xml_string(xml)

# XMLファイルとして書き出す
mujoco.mj_saveLastXML("arm.xml", model)