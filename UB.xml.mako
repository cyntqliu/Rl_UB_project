<%
	from rllab.misc.mako_utils import compute_rect_vertices
	ring_friction = 0.0005
	eu_friction = 0.0005
	eu_height = 0.1
	ring_radius = .5
%>

<box2d>
	<world timestep="0.05" gravity="0,0">
		<body name="ring" type="dynamic" position="0,${ring_radius}">
			<fixture
			  density="1"
			  friction="${ring_friction}"
			  shape="circle"
			  radius="${ring_radius}"
			/>
		</body>
		<body name="eu_cradle" type="dynamic" position="${ring_radius},${ring_radius}">
			<fixture
			  density="1"
			  friction="${ring_friction}"
			  shape="polygon"
			  vertices="${compute_rect_vertices((0,0),(0,${eu_height}),eu_height/20}"
			/>
		</body>
		
		<body name="x_track" type="static" position="0,${ring_radius}">
			<fixture friction="${ring_friction}" group="-1" shape="polygon" box="180,0.1"/>
		</body>
		<body name="y_track" type="dynamic" position="0,.505">
			<fixture friction="${eu_friction}" group="-1" shape="polygon" box="0.1,180"/>
		</body>
		
		<joint type="prisimatic" name="axis_slide" bodyA="x_track" bodyB="y_track"/>
		<joint type="prisimatic" name="xaxis" bodyA="x_track" bodyB="ring"/>
		<joint type="prisimatic" name="yaxis" bodyA="y_track" bodyB="eu_cradle"/>
		<state type="xpos" body="ring"/>
		<state type="ypos" body="eu_cradle"/>
		
	</world>
</box2d>