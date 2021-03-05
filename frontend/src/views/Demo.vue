<template>
  <section class="demo page_frame">
    <div class="header_wrapper">
      <header class="title_wrapper">
        <h1 class="title">Demo</h1>
        <div class="comment">Try to upload a facial image</div>
      </header>
    </div>
    <Uploader v-if="!image" accept="image/*" :multiple="false" @uploaded="getImage" />
    <DemoOptions v-else-if="!result" :image="image" @next="process" />
    <DemoResult v-else :data="result" />
  </section>
</template>

<script>
import Uploader from '@/components/Uploader.vue';
import DemoOptions from '@/components/DemoOptions.vue';
import DemoResult from '@/components/DemoResult.vue';

import { demoProcess } from '@/service';

export default {
  data() {
    return {
      image: null,
      result: null,
    };
  },
  methods: {
    /**
     * 从 Uploader 获取上传的图片数据
     */
    getImage(file) {
      this.image = {
        ...file,
        width: 0,
        height: 0,
      };
      // 获取图片的分辨率
      const image = new Image();
      image.src = file.base64;
      if (image.complete) {
        // 如果有缓存则读缓存
        this.image.width = image.width;
        this.image.height = image.height;
      } else {
        // 没缓存则需要加载一次
        image.onload = () => {
          this.image.width = image.width;
          this.image.height = image.height;
          image.onload = null;
        };
      }
    },
    async process() {
      try {
        const res = await demoProcess(this.image.rawFile);
        console.debug('demoProcess', res);
        this.result = res.data;
      } catch (e) {
        console.error('demo process error', e);
      }
    },
  },
  components: {
    Uploader,
    DemoOptions,
    DemoResult,
  },
};
</script>

<style lang="scss" scoped>
.demo {
}
</style>
