try:
    import Box2D
    from Box2D.b2 import contactListener, fixtureDef, polygonShape
except ImportError:
    raise DependencyNotInstalled("box2D is not installed, run `pip install gym[box2d]`")


        
        
 
class FrictionDetector(contactListener):
    def __init__(self, env, lap_complete_percent,
                 max_reward_tile:int=5000.):
        contactListener.__init__(self)
        self.env = env
        self.max_reward_tile = max_reward_tile
        self.lap_complete_percent = lap_complete_percent

    def BeginContact(self, contact):
        self._contact(contact, True)

    def EndContact(self, contact):
        self._contact(contact, False)
    
    def _contact(self, contact, begin):
        tile = None
        obj = None
        u1 = contact.fixtureA.body.userData
        u2 = contact.fixtureB.body.userData
        if u1 and "road_friction" in u1.__dict__:
            tile = u1
            obj = u2 #아마 여기서 obj라고 하면 바퀴일 것임.#
        if u2 and "road_friction" in u2.__dict__:
            tile = u2
            obj = u1
        if not tile:
            return

        # inherit tile color from env
        tile.color[:] = self.env.road_color
        if not obj or "tiles" not in obj.__dict__:
            return
        if begin: 
            obj.tiles.add(tile)
            if not tile.road_visited:
                tile.road_visited = True
                # self.env.reward += 1000.0 / len(self.env.track)
                ## tile 방문 할 때마다 max_reward / len(self.env_track) 만큼 reward가 쌓이기 때문에,
                # 만약에 전체 track을 통과를 한다면 max_reward만큼의 점수를 얻을 수 있게 된다
                if self.env.input_args.with_tile_reward:
                    self.env.reward += float(self.max_reward_tile) / len(self.env.track)
                self.env.tile_visited_count += 1

                # Lap is considered completed if enough % of the track was covered
                if (
                    tile.idx == 0
                    and self.env.tile_visited_count / len(self.env.track)
                    > self.lap_complete_percent
                ):
                    self.env.new_lap = True

            else: #동일한 tile에 또 계속 위치하는 경우 -> 속도가 안난다는 것이기 때문에 penalty 제공#
                tile.num_visited += 1
                self.env.same_tile_count = tile.num_visited
                if tile.num_visited >= self.env.input_args.same_tile_penalty:
                    self.env.on_same_tile_penalty = True
                

        else:
            obj.tiles.remove(tile)

 

class TileLocation_Detector(contactListener):
    # super-super class is contactListener
    def __init__(self, env, lap_complete_percent, max_reward_tile):
        contactListener.__init__(self)
        self.env = env
        self.max_reward_tile = max_reward_tile
        self.lap_complete_percent = lap_complete_percent

    def BeginContact(self, contact):
        self._contact(contact, True)

    def EndContact(self, contact):
        self._contact(contact, False)
        
    def _oscillation_penalty(self):
        dyna_obj = self.env.dynamics_obj
        e_c_arr = dyna_obj.e_c_arr
        if len(e_c_arr) == 1:
            return False, -1
        prev, cur = e_c_arr[-2], e_c_arr[-1]
        if prev * cur >= 0:
            return True, abs(prev - cur)
        else:
            if prev > cur:
                return True, prev - cur
            else:
                return True, cur - prev
            
    def _contact(self, contact, begin):
        tile = None
        obj = None
        u1 = contact.fixtureA.body.userData
        u2 = contact.fixtureB.body.userData
        
        if u1 and 'road_friction' in u1.__dict__:
            tile, obj = u1, u2 # road tile <-> wheel #
        if u2 and 'road_friction' in u2.__dict__:
            tile, obj = u2, u1
        if not tile:
            return
        ## tire object와 grass 영역이 하나라도 닿았을때 고정된 penalty를 부여할 수 있도록??
        ### 그렇다면 tile이 road가 아닌 grass임을 알 수 있을까?
        tile.color[:] = self.env.road_color
        if not obj or "tiles" not in obj.__dict__: # no contact occurred
            return
        if begin: 
            obj.tiles.add(tile) 
            if not tile.road_visited:
                # breakpoint()
                single_tile_reward = float(self.max_reward_tile) / len(self.env.track)
                if tile.is_straight: # 트랙에서의 직선 구간 통과
                    do_penalty, derivate_e_c = self._oscillation_penalty()
                    # breakpoint()
                    if not do_penalty:
                        self.env.reward += single_tile_reward
                    else:
                        # breakpoint()
                        ## track_width * 2 = 트랙 전체의 너비이기 때문에 2를 곱해주어야 함.
                        k = derivate_e_c / (self.env.track_width * 2) 
                        k = 1. - k
                        ## oscillation_penalty weight는 최대가 1이고, 더 작아짐에 따라서 oscillating에 대한, 즉 e_c의 미분에 대한 penalty의 가중치를 줄이는 것이다.
                        # 우선 최대는 1로
                        if self.env.input_args.do_penalty_max_reward:
                            ## 이렇게 하면 reward += single_tile_reward * (1-k)그 자체이고, 0 <= k <= 1이기 때문에 되게 작은 양으로 영향을 줌
                            # 그럼 track 가장자리만 따라서 이동하는 것은 줄어들 수 있으며, 약간의 oscillation에 대해서는 OK일듯
                            # 이런 식으로 하면 tile 한칸 이동하는 것에 대해서 최대 reward는 항상 1로 고정!!
                            k /= single_tile_reward
                        else:   ## default value for oscillation_penalty is 1
                            k *= self.env.input_args.oscillation_penalty
                        self.env.reward += single_tile_reward * k
                else:
                    self.env.reward += single_tile_reward
                tile.road_visited = True
                self.env.tile_visited_count += 1
                if (
                    tile.idx == 0 and self.env.tile_visited_count / len(self.env.track) > self.lap_complete_percent
                ):
                    self.env.new_lap = True
        else:
            obj.tiles.remove(tile)
            
class PenaltyDetector(FrictionDetector):
    def __init__(self, env, lap_complete_percent, max_reward_tile):
        super(PenaltyDetector, self).__init__(env=env, lap_complete_percent=lap_complete_percent,
                                              max_reward_tile=max_reward_tile)
    
    def BeginContact(self, contact):
        self._contact(contact, True)
        self._contact_limit(contact, True)
    
    def EndContact(self, contact):
        self._contact(contact, False)
        
    def _contact_limit(self, contact, begin):
        limit_tile, obj = None, None
        u1 = contact.fixtureA.body.userData
        u2 = contact.fixtureB.body.userData
        
        if u1 and "road_friction" in u1.__dict__ and "name" in u1.__dict__:
            limit_tile = u1;obj = u2
        
        if u2 and "road_friction" in u2.__dict__ and "name" in u2.__dict__:
            limit_tile = u2;obj = u1
        
        ## 경계 타일과 접촉한 object가 없거나, object가 차량의 바퀴가 아니라면 return ##
        if not obj or "limit_tiles" not in obj.__dict__:
            return
        
        if not limit_tile:
            return
        
        if begin:
            obj.limit_tiles.add(limit_tile)
            limit_tile.limit_visit_count += 1
 
            self.env.car_left_track = True
            
            