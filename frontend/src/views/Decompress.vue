<template>
  <section class="decompressor page_frame">
    <div class="header_wrapper">
      <header class="title_wrapper">
        <h1 class="title">Decompressor</h1>
        <div class="comment">Upload compressed images to decompress</div>
      </header>
      <button v-if="result" class="action_btn shadow_s_deep clickable" @click="downloadAll">
        DOWNLOAD ALL
      </button>
      <button
        v-else-if="fileList.length"
        class="action_btn shadow_s_deep clickable"
        @click="upload"
      >
        UPLOAD
      </button>
    </div>
    <Uploader type="fic" v-model="fileList" />
  </section>
</template>

<script>
import Uploader from '@/components/Uploader.vue';

import { decompress } from '@/service';

export default {
  data() {
    return {
      fileList: [],
      result: null,
    };
  },
  methods: {
    async upload() {
      try {
        const files = this.fileList.map((item) => item.rawFile);
        this.result = await decompress(files);
        console.debug('decompress result', this.result);
        for (let i = 0; i < this.fileList.length; i += 1) {
          this.fileList[i].result = this.result[i];
        }
        // Vue不会监听数组对象的更改，故需要调用方法手动触发Vue的监听
        this.fileList.splice(0, 0);
      } catch (e) {
        console.error('decompress error', e);
      }
    },
    downloadAll() {
      // TODO
    },
  },
  components: {
    Uploader,
  },
};
</script>

<style lang="scss" scoped>
.decompressor {
}
</style>
