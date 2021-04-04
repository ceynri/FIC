<template>
  <section class="demo page_frame">
    <div class="header_wrapper">
      <header class="title_wrapper">
        <h1 class="title">Demo</h1>
        <div class="comment" v-if="state === 1">Try to upload a facial image</div>
        <div class="comment" v-else-if="state === 2">
          Select the appropriate compression options
        </div>
        <div class="comment" v-else>
          Hover the cursor over the image below to compare with the original image
        </div>
      </header>
    </div>
    <Uploader v-if="state === 1" v-model="fileList" type="image" :multiple="false" />
    <DemoOptions v-else-if="state === 2" :image="image" @next="process" />
    <DemoResult v-else :data="result" />
  </section>
</template>

<script>
import Uploader from '@/components/Uploader.vue';
import DemoOptions from '@/components/DemoOptions.vue';
import DemoResult from '@/components/DemoResult.vue';

// import { demoProcess } from '@/service';

export default {
  data() {
    return {
      image: null,
      result: null,
      fileList: [],
    };
  },
  computed: {
    state() {
      if (!this.image) {
        return 1;
      }
      if (!this.result) {
        return 2;
      }
      return 3;
    },
  },
  watch: {
    /**
     * 从 Uploader 获取上传的图片数据
     */
    fileList() {
      const file = this.fileList[0];
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
  },
  methods: {
    async process() {
      try {
        this.result = await demoProcess(this.image.rawFile);
        console.debug('demoProcess', this.result);
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
