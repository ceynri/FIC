<template>
  <section class="compressor page_frame">
    <div class="header_wrapper">
      <header class="title_wrapper">
        <h1 class="title">Compressor</h1>
        <div class="comment">Upload facial images to compress</div>
      </header>
      <button v-if="true" class="action_btn shadow_s_deep clickable" @click="upload">UPLOAD</button>
      <button v-else class="action_btn shadow_s_deep clickable" @click="downloadAll">
        DOWNLOAD ALL
      </button>
    </div>
    <Uploader accept="image/*" v-model="fileList" />
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
        console.debug(files);
        this.result = await compress(files);
        console.debug('compress', this.result);
      } catch (e) {
        console.error('compress error', e);
      }
    },
    downloadAll() {},
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
