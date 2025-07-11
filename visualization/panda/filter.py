# 继承direct.filter.CommonFilters类进行细微调整
# author: weiwei
# date: 20201210


from direct.filter.CommonFilters import *
from direct.task.TaskManagerGlobal import taskMgr
import visualization.panda.filtermanager as fm

# 定义着色器代码片段,用于卡通效果和环境光遮蔽等
CARTOON_BODY_1 = """
float4 cartoondelta_n = k_cartoonseparation * texpix_txaux.xwyw;
float4 cartoon_n_ref = tex2D(k_txcolor, %(texcoord)s);
float4 cartoon_n0 = tex2D(k_txaux, %(texcoord)s + cartoondelta_n.xy);
float4 cartoon_n1 = tex2D(k_txaux, %(texcoord)s - cartoondelta_n.xy);
float4 cartoon_n2 = tex2D(k_txaux, %(texcoord)s + cartoondelta_n.wz);
float4 cartoon_n3 = tex2D(k_txaux, %(texcoord)s - cartoondelta_n.wz);
float4 cartoon_n_mx = max(cartoon_n0, max(cartoon_n1, max(cartoon_n2, cartoon_n3)));
float4 cartoon_n_mn = min(cartoon_n0, min(cartoon_n1, min(cartoon_n2, cartoon_n3)));
//float4 cartoon_n4 = tex2D(k_txaux, %(texcoord)s + cartoondelta_n.xz);
//float4 cartoon_n5 = tex2D(k_txaux, %(texcoord)s - cartoondelta_n.xz);
//float4 cartoon_n6 = tex2D(k_txaux, %(texcoord)s + float2(cartoondelta_n.x, -cartoondelta_n.z));
//float4 cartoon_n7 = tex2D(k_txaux, %(texcoord)s - float2(cartoondelta_n.x, -cartoondelta_n.z));
//float4 cartoon_n_mx = max(cartoon_n0, max(cartoon_n1, max(cartoon_n2, max(cartoon_n3, max(cartoon_n4, max(cartoon_n5, 
//max(cartoon_n6, cartoon_n7)))))));
//float4 cartoon_n_mn = min(cartoon_n0, min(cartoon_n1, min(cartoon_n2, min(cartoon_n3, min(cartoon_n4, min(cartoon_n5, 
//min(cartoon_n6, cartoon_n7)))))));
float cartoon_n_thresh = saturate(dot(cartoon_n_mx - cartoon_n_mn, float4(3,3,0,0)) - 0.5);
"""

CARTOON_BODY_2 = """
float4 cartoondelta_c = k_cartoonseparation * texpix_txcolor.xwyw;
float4 cartoon_c_ref = tex2D(k_txcolor, %(texcoord)s);
float4 cartoon_c0 = tex2D(k_txcolor, %(texcoord)s + cartoondelta_c.xy);
float4 cartoon_c1 = tex2D(k_txcolor, %(texcoord)s - cartoondelta_c.xy);
float4 cartoon_c2 = tex2D(k_txcolor, %(texcoord)s + cartoondelta_c.wz);
float4 cartoon_c3 = tex2D(k_txcolor, %(texcoord)s - cartoondelta_c.wz);
float4 cartoon_c_mx = max(cartoon_c0, max(cartoon_c1, max(cartoon_c2, cartoon_c3)));
float4 cartoon_c_mn = min(cartoon_c0, min(cartoon_c1, min(cartoon_c2, cartoon_c3)));
//float4 cartoon_c4 = tex2D(k_txcolor, %(texcoord)s + cartoondelta_c.xz);
//float4 cartoon_c5 = tex2D(k_txcolor, %(texcoord)s - cartoondelta_c.xz);
//float4 cartoon_c6 = tex2D(k_txcolor, %(texcoord)s + float2(cartoondelta_c.x, -cartoondelta_c.z));
//float4 cartoon_c7 = tex2D(k_txcolor, %(texcoord)s - float2(cartoondelta_c.x, -cartoondelta_c.z));
//float4 cartoon_c_mx = max(cartoon_c0, max(cartoon_c1, max(cartoon_c2, max(cartoon_c3, max(cartoon_c4, max(cartoon_c5, 
//max(cartoon_c6, cartoon_c7)))))));
//float4 cartoon_c_mn = min(cartoon_c0, min(cartoon_c1, min(cartoon_c2, min(cartoon_c3, min(cartoon_c4, min(cartoon_c5, 
//min(cartoon_c6, cartoon_c7)))))));
float cartoon_c_thresh = saturate(dot(cartoon_c_mx - cartoon_c_mn, float4(3,3,0,0)) - 0.5);
float cartoon_thresh = saturate(cartoon_n_thresh + cartoon_c_thresh);
o_color = lerp(o_color, k_cartooncolor, cartoon_thresh);
"""

# 某些 GPU 不支持可变长度循环
# 在配置着色器时,我们会在循环限制中填写 numsamples 的实际值


SSAO_BODY = """//Cg

void vshader(float4 vtx_position : POSITION,
             out float4 l_position : POSITION,
             out float2 l_texcoord : TEXCOORD0,
             out float2 l_texcoordD : TEXCOORD1,
             out float2 l_texcoordN : TEXCOORD2,
             uniform float4 texpad_depth,
             uniform float4 texpad_normal,
             uniform float4x4 mat_modelproj)
{
  l_position = mul(mat_modelproj, vtx_position);
  l_texcoord = vtx_position.xz;
  l_texcoordD = (vtx_position.xz * texpad_depth.xy) + texpad_depth.xy;
  l_texcoordN = (vtx_position.xz * texpad_normal.xy) + texpad_normal.xy;
}

float3 sphere[16] = float3[](float3(0.53812504, 0.18565957, -0.43192),float3(0.13790712, 0.24864247, 0.44301823),float3(0.33715037, 0.56794053, -0.005789503),float3(-0.6999805, -0.04511441, -0.0019965635),float3(0.06896307, -0.15983082, -0.85477847),float3(0.056099437, 0.006954967, -0.1843352),float3(-0.014653638, 0.14027752, 0.0762037),float3(0.010019933, -0.1924225, -0.034443386),float3(-0.35775623, -0.5301969, -0.43581226),float3(-0.3169221, 0.106360726, 0.015860917),float3(0.010350345, -0.58698344, 0.0046293875),float3(-0.08972908, -0.49408212, 0.3287904),float3(0.7119986, -0.0154690035, -0.09183723),float3(-0.053382345, 0.059675813, -0.5411899),float3(0.035267662, -0.063188605, 0.54602677),float3(-0.47761092, 0.2847911, -0.0271716));

void fshader(out float4 o_color : COLOR,
             uniform float4 k_params1,
             uniform float4 k_params2,
             float2 l_texcoord : TEXCOORD0,
             float2 l_texcoordD : TEXCOORD1,
             float2 l_texcoordN : TEXCOORD2,
             uniform sampler2D k_random : TEXUNIT0,
             uniform sampler2D k_depth : TEXUNIT1,
             uniform sampler2D k_normal : TEXUNIT2)
{
  float pixel_depth = tex2D(k_depth, l_texcoordD).a;
  float3 pixel_normal = (tex2D(k_normal, l_texcoordN).xyz * 2.0 - 1.0);
  float3 random_vector = normalize((tex2D(k_random, l_texcoord * 18.0 + pixel_depth + pixel_normal.xy).xyz * 2.0) - float3(1.0)).xyz;
  float occlusion = 0.0;
  float radius = k_params1.z / pixel_depth;
  float depth_difference;
  float3 sample_normal;
  float3 ray;
  for(int i = 0; i < %d; ++i) {
   ray = radius * reflect(sphere[i], random_vector);
   sample_normal = (tex2D(k_normal, l_texcoordN + ray.xy).xyz * 2.0 - 1.0);
   depth_difference =  (pixel_depth - tex2D(k_depth,l_texcoordD + ray.xy).r);
   occlusion += step(k_params2.y, depth_difference) * (1.0 - dot(sample_normal.xyz, pixel_normal)) * (1.0 - smoothstep(k_params2.y, k_params2.x, depth_difference));
  }
  o_color.rgb = 1.0 + (occlusion * k_params1.y);
  o_color.a = 1.0;
}
"""


class Filter(CommonFilters):

    def __init__(self, win, cam):
        """
        初始化 Filter 对象

        :param win: 渲染窗口
        :param cam: 摄像机对象
        """
        super().__init__(win, cam)
        self.manager = fm.FilterManager(win, cam)

    def reconfigure(self, fullrebuild, changed):
        """
        重新配置滤镜设置

        :param fullrebuild: 是否完全重建滤镜
        :param changed: 改变的滤镜名称
        :return: 是否成功重新配置
        """
        configuration = self.configuration
        if (fullrebuild):
            self.cleanup()
            if (len(configuration) == 0):
                return
            if not self.manager.win.gsg.getSupportsBasicShaders():
                return False
            auxbits = 0
            needtex = set(["color"])
            needtexcoord = set(["color"])
            if ("CartoonInk" in configuration):
                needtex.add("aux")
                auxbits |= AuxBitplaneAttrib.ABOAuxNormal
                needtexcoord.add("aux")
            if ("AmbientOcclusion" in configuration):
                needtex.add("depth")
                needtex.add("ssao0")
                needtex.add("ssao1")
                needtex.add("ssao2")
                needtex.add("aux")
                auxbits |= AuxBitplaneAttrib.ABOAuxNormal
                needtexcoord.add("ssao2")
            if ("BlurSharpen" in configuration):
                needtex.add("blur0")
                needtex.add("blur1")
                needtexcoord.add("blur1")
            if ("Bloom" in configuration):
                needtex.add("bloom0")
                needtex.add("bloom1")
                needtex.add("bloom2")
                needtex.add("bloom3")
                auxbits |= AuxBitplaneAttrib.ABOGlow
                needtexcoord.add("bloom3")
            if ("ViewGlow" in configuration):
                auxbits |= AuxBitplaneAttrib.ABOGlow
            if ("VolumetricLighting" in configuration):
                needtex.add(configuration["VolumetricLighting"].source)
            for tex in needtex:
                self.textures[tex] = Texture("scene-" + tex)
                self.textures[tex].setWrapU(Texture.WMClamp)
                self.textures[tex].setWrapV(Texture.WMClamp)
            fbprops = None
            clamping = None
            if "HighDynamicRange" in configuration:
                fbprops = FrameBufferProperties()
                fbprops.setFloatColor(True)
                fbprops.setSrgbColor(False)
                clamping = False
            self.finalQuad = self.manager.renderSceneInto(textures=self.textures, auxbits=auxbits, fbprops=fbprops,
                                                          clamping=clamping)
            if (self.finalQuad == None):
                self.cleanup()
                return False
            if ("BlurSharpen" in configuration):
                blur0 = self.textures["blur0"]
                blur1 = self.textures["blur1"]
                self.blur.append(self.manager.renderQuadInto("filter-blur0", colortex=blur0, div=2))
                self.blur.append(self.manager.renderQuadInto("filter-blur1", colortex=blur1))
                self.blur[0].setShaderInput("src", self.textures["color"])
                self.blur[0].setShader(Shader.make(BLUR_X, Shader.SL_Cg))
                self.blur[1].setShaderInput("src", blur0)
                self.blur[1].setShader(Shader.make(BLUR_Y, Shader.SL_Cg))
            if ("AmbientOcclusion" in configuration):
                ssao0 = self.textures["ssao0"]
                ssao1 = self.textures["ssao1"]
                ssao2 = self.textures["ssao2"]
                self.ssao.append(self.manager.renderQuadInto("filter-ssao0", colortex=ssao0))
                self.ssao.append(self.manager.renderQuadInto("filter-ssao1", colortex=ssao1, div=2))
                self.ssao.append(self.manager.renderQuadInto("filter-ssao2", colortex=ssao2))
                self.ssao[0].setShaderInput("depth", self.textures["depth"])
                self.ssao[0].setShaderInput("normal", self.textures["aux"])
                self.ssao[0].setShaderInput("random", loader.loadTexture("maps/random.rgb"))
                self.ssao[0].setShader(
                    Shader.make(SSAO_BODY % configuration["AmbientOcclusion"].numsamples, Shader.SL_Cg))
                self.ssao[1].setShaderInput("src", ssao0)
                self.ssao[1].setShader(Shader.make(BLUR_X, Shader.SL_Cg))
                self.ssao[2].setShaderInput("src", ssao1)
                self.ssao[2].setShader(Shader.make(BLUR_Y, Shader.SL_Cg))
            if ("Bloom" in configuration):
                bloomconf = configuration["Bloom"]
                bloom0 = self.textures["bloom0"]
                bloom1 = self.textures["bloom1"]
                bloom2 = self.textures["bloom2"]
                bloom3 = self.textures["bloom3"]
                if (bloomconf.size == "large"):
                    scale = 8
                    downsamplerName = "filter-down4"
                    downsampler = DOWN_4
                elif (bloomconf.size == "medium"):
                    scale = 4
                    downsamplerName = "filter-copy"
                    downsampler = COPY
                else:
                    scale = 2
                    downsamplerName = "filter-copy"
                    downsampler = COPY
                self.bloom.append(self.manager.renderQuadInto("filter-bloomi", colortex=bloom0, div=2, align=scale))
                self.bloom.append(self.manager.renderQuadInto(downsamplerName, colortex=bloom1, div=scale, align=scale))
                self.bloom.append(self.manager.renderQuadInto("filter-bloomx", colortex=bloom2, div=scale, align=scale))
                self.bloom.append(self.manager.renderQuadInto("filter-bloomy", colortex=bloom3, div=scale, align=scale))
                self.bloom[0].setShaderInput("src", self.textures["color"])
                self.bloom[0].setShader(Shader.make(BLOOM_I, Shader.SL_Cg))
                self.bloom[1].setShaderInput("src", bloom0)
                self.bloom[1].setShader(Shader.make(downsampler, Shader.SL_Cg))
                self.bloom[2].setShaderInput("src", bloom1)
                self.bloom[2].setShader(Shader.make(BLOOM_X, Shader.SL_Cg))
                self.bloom[3].setShaderInput("src", bloom2)
                self.bloom[3].setShader(Shader.make(BLOOM_Y, Shader.SL_Cg))
            texcoords = {}
            texcoordPadding = {}
            for tex in needtexcoord:
                if self.textures[tex].getAutoTextureScale() != ATSNone or \
                        "HalfPixelShift" in configuration:
                    texcoords[tex] = "l_texcoord_" + tex
                    texcoordPadding["l_texcoord_" + tex] = tex
                else:
                    # 分享未填充的纹理坐标
                    texcoords[tex] = "l_texcoord"
                    texcoordPadding["l_texcoord"] = None
            texcoordSets = list(enumerate(texcoordPadding.keys()))
            text = "//Cg\n"
            if "HighDynamicRange" in configuration:
                text += "static const float3x3 aces_input_mat = {\n"
                text += "  {0.59719, 0.35458, 0.04823},\n"
                text += "  {0.07600, 0.90834, 0.01566},\n"
                text += "  {0.02840, 0.13383, 0.83777},\n"
                text += "};\n"
                text += "static const float3x3 aces_output_mat = {\n"
                text += "  { 1.60475, -0.53108, -0.07367},\n"
                text += "  {-0.10208,  1.10813, -0.00605},\n"
                text += "  {-0.00327, -0.07276,  1.07602},\n"
                text += "};\n"

            # 顶点着色器
            text += "void vshader(float4 vtx_position : POSITION,\n"
            text += "  uniform float4x4 mat_modelproj,\n"
            for texcoord, padTex in texcoordPadding.items():
                if padTex is not None:
                    text += "  uniform float4 texpad_tx%s,\n" % (padTex)
                    if ("HalfPixelShift" in configuration):
                        text += "  uniform float4 texpix_tx%s,\n" % (padTex)
            for i, name in texcoordSets:
                text += "  out float2 %s : TEXCOORD%d,\n" % (name, i)
            text += "  out float4 l_position : POSITION)"
            text += "{\n"
            text += "  l_position = mul(mat_modelproj, vtx_position);\n"
            for texcoord, padTex in texcoordPadding.items():
                if padTex is None:
                    text += "  %s = vtx_position.xz * float2(0.5, 0.5) + float2(0.5, 0.5);\n" % (texcoord)
                else:
                    text += "  %s = (vtx_position.xz * texpad_tx%s.xy) + texpad_tx%s.xy;\n" % (texcoord, padTex, padTex)
                    if ("HalfPixelShift" in configuration):
                        text += "  %s += texpix_tx%s.xy * 0.5;\n" % (texcoord, padTex)
            text += "}\n"

            # 片段着色器
            text += "void fshader(\n"
            for i, name in texcoordSets:
                text += "  float2 %s : TEXCOORD%d,\n" % (name, i)
            for key in self.textures:
                text += "  uniform sampler2D k_tx" + key + ",\n"
            if ("CartoonInk" in configuration):
                text += "  uniform float4 k_cartoonseparation,\n"
                text += "  uniform float4 k_cartooncolor,\n"
                text += "  uniform float4 texpix_txaux,\n"
                text += "  uniform float4 texpix_txcolor,\n"
            if ("BlurSharpen" in configuration):
                text += "  uniform float4 k_blurval,\n"
            if ("VolumetricLighting" in configuration):
                text += "  uniform float4 k_casterpos,\n"
                text += "  uniform float4 k_vlparams,\n"
            if ("ExposureAdjust" in configuration):
                text += "  uniform float k_exposure,\n"
            text += "  out float4 o_color : COLOR)\n"
            text += "{\n"
            text += "  o_color = tex2D(k_txcolor, %s);\n" % (texcoords["color"])
            if ("CartoonInk" in configuration):
                text += CARTOON_BODY_1 % {"texcoord": texcoords["aux"]}
                text += CARTOON_BODY_2 % {"texcoord": texcoords["color"]}
            if ("AmbientOcclusion" in configuration):
                text += "  o_color *= tex2D(k_txssao2, %s).r;\n" % (texcoords["ssao2"])
            if ("BlurSharpen" in configuration):
                text += "  o_color = lerp(tex2D(k_txblur1, %s), o_color, k_blurval.x);\n" % (texcoords["blur1"])
            if ("Bloom" in configuration):
                text += "  o_color = saturate(o_color);\n";
                text += "  float4 bloom = 0.5 * tex2D(k_txbloom3, %s);\n" % (texcoords["bloom3"])
                text += "  o_color = 1-((1-bloom)*(1-o_color));\n"
            if ("ViewGlow" in configuration):
                text += "  o_color.r = o_color.a;\n"
            if ("VolumetricLighting" in configuration):
                text += "  float decay = 1.0f;\n"
                text += "  float2 curcoord = %s;\n" % (texcoords["color"])
                text += "  float2 lightdir = curcoord - k_casterpos.xy;\n"
                text += "  lightdir *= k_vlparams.x;\n"
                text += "  half4 sample = tex2D(k_txcolor, curcoord);\n"
                text += "  float3 vlcolor = sample.rgb * sample.a;\n"
                text += "  for (int i = 0; i < %s; i++) {\n" % (int(configuration["VolumetricLighting"].numsamples))
                text += "    curcoord -= lightdir;\n"
                text += "    sample = tex2D(k_tx%s, curcoord);\n" % (configuration["VolumetricLighting"].source)
                text += "    sample *= sample.a * decay;//*weight\n"
                text += "    vlcolor += sample.rgb;\n"
                text += "    decay *= k_vlparams.y;\n"
                text += "  }\n"
                text += "  o_color += float4(vlcolor * k_vlparams.z, 1);\n"
            if ("ExposureAdjust" in configuration):
                text += "  o_color.rgb *= k_exposure;\n"

            # 感谢 Stephen Hill！
            if ("HighDynamicRange" in configuration):
                text += "  float3 aces_color = mul(aces_input_mat, o_color.rgb);\n"
                text += "  o_color.rgb = saturate(mul(aces_output_mat, (aces_color * (aces_color + 0.0245786f) - 0.000090537f) / (aces_color * (0.983729f * aces_color + 0.4329510f) + 0.238081f)));\n"
            if ("GammaAdjust" in configuration):
                gamma = configuration["GammaAdjust"]
                if gamma == 0.5:
                    text += "  o_color.rgb = sqrt(o_color.rgb);\n"
                elif gamma == 2.0:
                    text += "  o_color.rgb *= o_color.rgb;\n"
                elif gamma != 1.0:
                    text += "  o_color.rgb = pow(o_color.rgb, %ff);\n" % (gamma)
            if ("SrgbEncode" in configuration):
                text += "  o_color.r = (o_color.r < 0.0031308) ? (o_color.r * 12.92) : (1.055 * pow(o_color.r, 0.41666) - 0.055);\n"
                text += "  o_color.g = (o_color.g < 0.0031308) ? (o_color.g * 12.92) : (1.055 * pow(o_color.g, 0.41666) - 0.055);\n"
                text += "  o_color.b = (o_color.b < 0.0031308) ? (o_color.b * 12.92) : (1.055 * pow(o_color.b, 0.41666) - 0.055);\n"
            if ("Inverted" in configuration):
                text += "  o_color = float4(1, 1, 1, 1) - o_color;\n"
            text += "}\n"
            shader = Shader.make(text, Shader.SL_Cg)
            if not shader:
                return False
            self.finalQuad.setShader(shader)
            for tex in self.textures:
                self.finalQuad.setShaderInput("tx" + tex, self.textures[tex])
            self.task = taskMgr.add(self.update, "common-filters-update")

        if (changed == "CartoonInk") or fullrebuild:
            if ("CartoonInk" in configuration):
                c = configuration["CartoonInk"]
                self.finalQuad.setShaderInput("cartoonseparation", LVecBase4(c.separation, 0, c.separation, 0))
                self.finalQuad.setShaderInput("cartooncolor", c.color)

        if (changed == "BlurSharpen") or fullrebuild:
            if ("BlurSharpen" in configuration):
                blurval = configuration["BlurSharpen"]
                self.finalQuad.setShaderInput("blurval", LVecBase4(blurval, blurval, blurval, blurval))

        if (changed == "Bloom") or fullrebuild:
            if ("Bloom" in configuration):
                bloomconf = configuration["Bloom"]
                intensity = bloomconf.intensity * 3.0
                self.bloom[0].setShaderInput("blend", bloomconf.blendx, bloomconf.blendy, bloomconf.blendz,
                                             bloomconf.blendw * 2.0)
                self.bloom[0].setShaderInput("trigger", bloomconf.mintrigger,
                                             1.0 / (bloomconf.maxtrigger - bloomconf.mintrigger), 0.0, 0.0)
                self.bloom[0].setShaderInput("desat", bloomconf.desat)
                self.bloom[3].setShaderInput("intensity", intensity, intensity, intensity, intensity)

        if (changed == "VolumetricLighting") or fullrebuild:
            if ("VolumetricLighting" in configuration):
                config = configuration["VolumetricLighting"]
                tcparam = config.density / float(config.numsamples)
                self.finalQuad.setShaderInput("vlparams", tcparam, config.decay, config.exposure, 0.0)

        if (changed == "AmbientOcclusion") or fullrebuild:
            if ("AmbientOcclusion" in configuration):
                config = configuration["AmbientOcclusion"]
                self.ssao[0].setShaderInput("params1", config.numsamples, -float(config.amount) / config.numsamples,
                                            config.radius, 0)
                self.ssao[0].setShaderInput("params2", config.strength, config.falloff, 0, 0)
        if (changed == "ExposureAdjust") or fullrebuild:
            if ("ExposureAdjust" in configuration):
                stops = configuration["ExposureAdjust"]
                self.finalQuad.setShaderInput("exposure", 2 ** stops)
        self.update()
        return True

    def setCartoonInk(self, separation=1, color=(0, 0, 0, 1)):
        """
        设置卡通墨水效果

        :param separation: 墨水分离度
        :param color: 墨水颜色
        :return: 是否成功设置
        """
        fullrebuild = (("CartoonInk" in self.configuration) == False)
        newconfig = FilterConfig()
        newconfig.separation = separation
        newconfig.color = color
        self.configuration["CartoonInk"] = newconfig
        return self.reconfigure(fullrebuild, "CartoonInk")

    def delCartoonInk(self):
        """
        删除卡通墨水效果

        :return: 是否成功删除
        """
        if ("CartoonInk" in self.configuration):
            del self.configuration["CartoonInk"]
            return self.reconfigure(True, "CartoonInk")
        return True

    def setBloom(self, blend=(0.3, 0.4, 0.3, 0.0), mintrigger=0.6, maxtrigger=1.0, desat=0.6, intensity=1.0,
                 size="medium"):
        """
        设置 Bloom 效果

        :param blend: 混合参数
        :param mintrigger: 最小触发值
        :param maxtrigger: 最大触发值
        :param desat: 去饱和度
        :param intensity: 强度
        :param size: 大小
        :return: 是否成功设置
        """

        if (size == 0):
            size = "off"
        elif (size == 1):
            size = "small"
        elif (size == 2):
            size = "medium"
        elif (size == 3):
            size = "large"
        if (size == "off"):
            self.delBloom()
            return
        if (maxtrigger == None): maxtrigger = mintrigger + 0.8
        oldconfig = self.configuration.get("Bloom", None)
        fullrebuild = True
        if (oldconfig) and (oldconfig.size == size):
            fullrebuild = False
        newconfig = FilterConfig()
        (newconfig.blendx, newconfig.blendy, newconfig.blendz, newconfig.blendw) = blend
        newconfig.maxtrigger = maxtrigger
        newconfig.mintrigger = mintrigger
        newconfig.desat = desat
        newconfig.intensity = intensity
        newconfig.size = size
        self.configuration["Bloom"] = newconfig
        return self.reconfigure(fullrebuild, "Bloom")

    def delBloom(self):
        """
        删除 Bloom 效果

        :return: 是否成功删除
        """
        if ("Bloom" in self.configuration):
            del self.configuration["Bloom"]
            return self.reconfigure(True, "Bloom")
        return True

    def setHalfPixelShift(self):
        """
        设置半像素偏移

        :return: 是否成功设置
        """
        fullrebuild = (("HalfPixelShift" in self.configuration) == False)
        self.configuration["HalfPixelShift"] = 1
        return self.reconfigure(fullrebuild, "HalfPixelShift")

    def delHalfPixelShift(self):
        """
        删除半像素偏移

        :return: 是否成功删除
        """
        if ("HalfPixelShift" in self.configuration):
            del self.configuration["HalfPixelShift"]
            return self.reconfigure(True, "HalfPixelShift")
        return True

    def setViewGlow(self):
        """
        设置视图辉光效果

        :return: 是否成功设置
        """
        fullrebuild = (("ViewGlow" in self.configuration) == False)
        self.configuration["ViewGlow"] = 1
        return self.reconfigure(fullrebuild, "ViewGlow")

    def delViewGlow(self):
        """
        删除视图辉光效果

        :return: 是否成功删除
        """
        if ("ViewGlow" in self.configuration):
            del self.configuration["ViewGlow"]
            return self.reconfigure(True, "ViewGlow")
        return True

    def setInverted(self):
        """
        设置反转效果

        :return: 是否成功设置
        """
        fullrebuild = (("Inverted" in self.configuration) == False)
        self.configuration["Inverted"] = 1
        return self.reconfigure(fullrebuild, "Inverted")

    def delInverted(self):
        """
        删除反转效果

        :return: 是否成功删除
        """
        if ("Inverted" in self.configuration):
            del self.configuration["Inverted"]
            return self.reconfigure(True, "Inverted")
        return True

    def setVolumetricLighting(self, caster, numsamples=32, density=5.0, decay=0.1, exposure=0.1, source="color"):
        """
        设置体积光效果

        :param caster: 光源对象
        :param numsamples: 样本数量
        :param density: 密度
        :param decay: 衰减
        :param exposure: 曝光
        :param source: 光源类型
        :return: 是否成功设置
        """

        oldconfig = self.configuration.get("VolumetricLighting", None)
        fullrebuild = True
        if (oldconfig) and (oldconfig.source == source) and (oldconfig.numsamples == int(numsamples)):
            fullrebuild = False
        newconfig = FilterConfig()
        newconfig.caster = caster
        newconfig.numsamples = int(numsamples)
        newconfig.density = density
        newconfig.decay = decay
        newconfig.exposure = exposure
        newconfig.source = source
        self.configuration["VolumetricLighting"] = newconfig
        return self.reconfigure(fullrebuild, "VolumetricLighting")

    def delVolumetricLighting(self):
        """
        删除体积光效果

        :return: 是否成功删除
        """
        if ("VolumetricLighting" in self.configuration):
            del self.configuration["VolumetricLighting"]
            return self.reconfigure(True, "VolumetricLighting")
        return True

    def setBlurSharpen(self, amount=0.0):
        """
        设置模糊/锐化效果

        启用模糊/锐化滤镜.如果“amount”参数为 1.0,则无效.0.0 表示完全模糊,高于 1.0 的值会使图像更加锐化.

        :param amount: 效果强度
        :return: 是否成功设置
        """
        fullrebuild = (("BlurSharpen" in self.configuration) == False)
        self.configuration["BlurSharpen"] = amount
        return self.reconfigure(fullrebuild, "BlurSharpen")

    def delBlurSharpen(self):
        """
        删除模糊/锐化效果

        :return: 是否成功删除
        """
        if ("BlurSharpen" in self.configuration):
            del self.configuration["BlurSharpen"]
            return self.reconfigure(True, "BlurSharpen")
        return True

    def setAmbientOcclusion(self, numsamples=16, radius=0.05, amount=2.0, strength=0.01, falloff=0.000002):
        """
        设置环境光遮蔽效果

        :param numsamples: 样本数量
        :param radius: 半径
        :param amount: 效果强度
        :param strength: 强度
        :param falloff: 衰减
        :return: 是否成功设置
        """
        fullrebuild = (("AmbientOcclusion" in self.configuration) == False)

        if (not fullrebuild):
            fullrebuild = (numsamples != self.configuration["AmbientOcclusion"].numsamples)

        newconfig = FilterConfig()
        newconfig.numsamples = numsamples
        newconfig.radius = radius
        newconfig.amount = amount
        newconfig.strength = strength
        newconfig.falloff = falloff
        self.configuration["AmbientOcclusion"] = newconfig
        return self.reconfigure(fullrebuild, "AmbientOcclusion")

    def delAmbientOcclusion(self):
        """
        删除环境光遮蔽效果

        :return: 是否成功删除
        """
        if ("AmbientOcclusion" in self.configuration):
            del self.configuration["AmbientOcclusion"]
            return self.reconfigure(True, "AmbientOcclusion")
        return True

    def setGammaAdjust(self, gamma):
        """
        对图像应用额外的伽马校正 1.0 = 无校正

        :param gamma: 伽马值
        :return: 是否成功设置
        """
        old_gamma = self.configuration.get("GammaAdjust", 1.0)
        if old_gamma != gamma:
            self.configuration["GammaAdjust"] = gamma
            return self.reconfigure(True, "GammaAdjust")
        return True

    def delGammaAdjust(self):
        """
        删除伽马校正.

        :return: 是否成功删除
        """
        if ("GammaAdjust" in self.configuration):
            old_gamma = self.configuration["GammaAdjust"]
            del self.configuration["GammaAdjust"]
            return self.reconfigure((old_gamma != 1.0), "GammaAdjust")
        return True

    def setSrgbEncode(self, force=False):
        """
        设置 sRGB 编码
        :param force: 是否强制应用
        :return: 是否成功设置

        将逆 sRGB EOTF 应用于输出,除非窗口,已经具有 sRGB 帧缓冲区,在这种情况下,此滤镜将拒绝应用
        ,以防止意外重复应用.将 force 参数设置为 True 可强制在所有情况下应用此滤镜.版本添加:: 1.10.7
        """
        new_enable = force or not self.manager.win.getFbProperties().getSrgbColor()
        old_enable = self.configuration.get("SrgbEncode", False)
        if new_enable and not old_enable:
            self.configuration["SrgbEncode"] = True
            return self.reconfigure(True, "SrgbEncode")
        elif not new_enable and old_enable:
            del self.configuration["SrgbEncode"]
        return new_enable

    def delSrgbEncode(self):
        """
        删除 sRGB 编码

        :return: 是否成功删除
        """
        """ Reverses the effects of setSrgbEncode. """
        if ("SrgbEncode" in self.configuration):
            old_enable = self.configuration["SrgbEncode"]
            del self.configuration["SrgbEncode"]
            return self.reconfigure(old_enable, "SrgbEncode")
        return True

    def setHighDynamicRange(self):
        """
        设置高动态范围

        通过使用浮点帧缓冲区启用 HDR 渲染,禁用主场景的色彩钳制,并应用色调映射操作符 (ACES).
        可能还需要使用 setExposureAdjust 来执行场景的曝光补偿,具体取决于光照强度.版本添加:: 1.10.7

        :return: 是否成功设置
        """
        fullrebuild = (("HighDynamicRange" in self.configuration) is False)
        self.configuration["HighDynamicRange"] = 1
        return self.reconfigure(fullrebuild, "HighDynamicRange")

    def delHighDynamicRange(self):
        """
        删除高动态范围

        :return: 是否成功删除
        """
        if ("HighDynamicRange" in self.configuration):
            del self.configuration["HighDynamicRange"]
            return self.reconfigure(True, "HighDynamicRange")
        return True

    def setExposureAdjust(self, stops):
        """
        设置曝光调整

        设置相对曝光调整,以与渲染场景的结果相乘(以档位为单位).0 表示无调整,正值表示图像更亮
        与 HDR 结合使用时很有用,请参阅 setHighDynamicRange.版本添加:: 1.10.7

        :param stops: 曝光档位
        :return: 是否成功设置
        """

        old_stops = self.configuration.get("ExposureAdjust")
        if old_stops != stops:
            self.configuration["ExposureAdjust"] = stops
            return self.reconfigure(old_stops is None, "ExposureAdjust")
        return True

    def delExposureAdjust(self):
        """
        删除曝光调整

        :return: 是否成功删除
        """
        if ("ExposureAdjust" in self.configuration):
            del self.configuration["ExposureAdjust"]
            return self.reconfigure(True, "ExposureAdjust")
        return True

    # snake_case 别名: 
    del_cartoon_ink = delCartoonInk
    set_half_pixel_shift = setHalfPixelShift
    del_half_pixel_shift = delHalfPixelShift
    set_inverted = setInverted
    del_inverted = delInverted
    del_view_glow = delViewGlow
    set_volumetric_lighting = setVolumetricLighting
    set_bloom = setBloom
    set_view_glow = setViewGlow
    set_ambient_occlusion = setAmbientOcclusion
    set_cartoon_ink = setCartoonInk
    del_bloom = delBloom
    del_ambient_occlusion = delAmbientOcclusion
    set_blur_sharpen = setBlurSharpen
    del_blur_sharpen = delBlurSharpen
    del_volumetric_lighting = delVolumetricLighting
    set_gamma_adjust = setGammaAdjust
    del_gamma_adjust = delGammaAdjust
    set_srgb_encode = setSrgbEncode
    del_srgb_encode = delSrgbEncode
    set_exposure_adjust = setExposureAdjust
    del_exposure_adjust = delExposureAdjust
    set_high_dynamic_range = setHighDynamicRange
    del_high_dynamic_range = delHighDynamicRange
