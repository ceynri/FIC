<template>
  <section class="compressor page_frame">
    <div class="header_wrapper">
      <header class="title_wrapper">
        <h1 class="title">Compressor</h1>
        <div class="comment">Upload facial images to compress</div>
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
    <Uploader type="image" v-model="fileList" />
  </section>
</template>

<script>
import Uploader from '@/components/Uploader.vue';

import { compress } from '@/service';

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
        this.result = await compress(files);
        console.debug('compress result', this.result);
        for (let i = 0; i < this.fileList.length; i += 1) {
          this.fileList[i].result = this.result[i];
        }
        this.fileList.splice(0, 0);
      } catch (e) {
        console.error('compress error', e);
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
.compressor {
}
</style>
